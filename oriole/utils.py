#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-05-17
# @Author  : Shawn Shan (shansixiong@cs.uchicago.edu)
# @Link    : https://www.shawnshan.com/


import errno
import glob
import gzip
import json
import os
import pickle
import random
import shutil
import sys
import tarfile
import zipfile

import PIL
import six
from six.moves.urllib.error import HTTPError, URLError

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras

sys.stderr = stderr
import keras.backend as K
import numpy as np
import tensorflow as tf
from PIL import Image, ExifTags
from keras.layers import Dense, Activation
from keras.models import Model
from keras.preprocessing import image

from fawkes.align_face import align
from six.moves.urllib.request import urlopen

if sys.version_info[0] == 2:
    def urlretrieve(url, filename, reporthook=None, data=None):
        def chunk_read(response, chunk_size=8192, reporthook=None):
            content_type = response.info().get('Content-Length')
            total_size = -1
            if content_type is not None:
                total_size = int(content_type.strip())
            count = 0
            while True:
                chunk = response.read(chunk_size)
                count += 1
                if reporthook is not None:
                    reporthook(count, chunk_size, total_size)
                if chunk:
                    yield chunk
                else:
                    break

        response = urlopen(url, data)
        with open(filename, 'wb') as fd:
            for chunk in chunk_read(response, reporthook=reporthook):
                fd.write(chunk)
else:
    from six.moves.urllib.request import urlretrieve


def clip_img(X, preprocessing='raw'):
    X = reverse_preprocess(X, preprocessing)
    X = np.clip(X, 0.0, 255.0)
    X = preprocess(X, preprocessing)
    return X


def load_image(path):
    try:
        img = Image.open(path)
    except PIL.UnidentifiedImageError:
        return None
    except IsADirectoryError:
        return None

    try:
        info = img._getexif()
    except OSError:
        return None

    if info is not None:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = dict(img._getexif().items())
        if orientation in exif.keys():
            if exif[orientation] == 3:
                img = img.rotate(180, expand=True)
            elif exif[orientation] == 6:
                img = img.rotate(270, expand=True)
            elif exif[orientation] == 8:
                img = img.rotate(90, expand=True)
            else:
                pass
    img = img.convert('RGB')
    image_array = image.img_to_array(img)

    return image_array


def filter_image_paths_CLQ(image_paths,power = 20,mode='min',th = 0.002):
    print("Identify {} files in the directory".format(len(image_paths)))
    new_image_paths = []
    new_images = []
    numbers = power ###需要cloak的照片数量
    
    index = 0
    for p in image_paths:
        img = load_image(p)
        if img is None:
            print("{} is not an image file, skipped".format(p.split("/")[-1]))
            continue
        lt = p.split('/')
        if power == 1:
            lt[-3] = '{}_{}_{}{}_cloaks'.format(lt[-3],'fawkes',mode,int(th*1000))
        else:
            lt[-3] ='{}_{}_{}{}_cloaks'.format(lt[-3],'oriole',mode,int(th*1000))
        p = '/'.join(lt)
        [new_image_paths.append("{}_{}_original.{}".format(p.split('.')[0], (index + i) % numbers,p.split('.')[-1])) for i in range(numbers)]
        [new_images.append(img) for i in range(numbers)]
    print("Identify {} images in the directory and I will cloak {} photos for each".format(len(new_image_paths) // numbers , numbers))
    return new_image_paths, new_images    
    

def filter_image_paths(image_paths):
    print("Identify {} files in the directory".format(len(image_paths)))
    new_image_paths = []
    new_images = []
   
    for p in image_paths:
        img = load_image(p)
        if img is None:
            print("{} is not an image file, skipped")
            continue
        new_image_paths.append(p)
        new_images.append(img)
    print("Identify {} images in the directory".format(len(new_image_paths)))
    return new_image_paths, new_images


class Faces(object):
    def __init__(self, image_paths, loaded_images, aligner, verbose=1, eval_local=False, preprocessing=True,
                 no_align=False):
        self.image_paths = image_paths
        self.verbose = verbose
        self.no_align = no_align
        self.aligner = aligner
        self.org_faces = []
        self.cropped_faces = []
        self.cropped_faces_shape = []
        self.cropped_index = []
        self.callback_idx = []
        for i in range(0, len(loaded_images)):
            cur_img = loaded_images[i]
            p = image_paths[i]
            self.org_faces.append(cur_img)

            if eval_local:
                margin = 0
            else:
                margin = 0.7

            if not no_align:
                align_img = align(cur_img, self.aligner, margin=margin)
                if align_img is None:
                    print("Find 0 face(s)".format(p.split("/")[-1]))
                    continue

                cur_faces = align_img[0]
            else:
                cur_faces = [cur_img]

            cur_shapes = [f.shape[:-1] for f in cur_faces]

            cur_faces_square = []
            if verbose and not no_align:
                print("Find {} face(s) in {}".format(len(cur_faces), p.split("/")[-1]))             
    
            if eval_local:
                cur_faces = cur_faces[:1]

            for img in cur_faces:####img.shape = (1440,1080,3)
                if eval_local:
                    base = resize(img, (224, 224))
                else:
                    long_size = max([img.shape[1], img.shape[0]])
                    base = np.zeros((long_size, long_size, 3))
                    # import pdb
                    # pdb.set_trace()

                    base[0:img.shape[0], 0:img.shape[1], :] = img###base.shape = (1440,1440,3)
                cur_faces_square.append(base)

            cur_faces_square = [resize(f, (224, 224)) for f in cur_faces_square]###cur_faces_square[0].shape = (224,224,3)
            self.cropped_faces.extend(cur_faces_square)

            if not self.no_align:
                cur_index = align_img[1]

                self.cropped_faces_shape.extend(cur_shapes)
                self.cropped_index.extend(cur_index)
                self.callback_idx.extend([i] * len(cur_faces_square))

        if len(self.cropped_faces) == 0:
            return

        self.cropped_faces = np.array(self.cropped_faces)

        if preprocessing:
            self.cropped_faces = preprocess(self.cropped_faces, 'imagenet')

        self.cloaked_cropped_faces = None
        self.cloaked_faces = np.copy(self.org_faces)

    def get_faces(self):
        return self.cropped_faces

    def merge_faces(self, protected_images, original_images):
        if self.no_align:
            return np.clip(protected_images, 0.0, 255.0)

        self.cloaked_faces = np.copy(self.org_faces)###org_faces为完全的原图，而original_images则是压缩过的图片

        for i in range(len(self.cropped_faces)):
            cur_protected = protected_images[i]
            cur_original = original_images[i]

            org_shape = self.cropped_faces_shape[i]####每张图片对应的shape
            old_square_shape = max([org_shape[0], org_shape[1]])

            cur_protected = resize(cur_protected, (old_square_shape, old_square_shape))
            cur_original = resize(cur_original, (old_square_shape, old_square_shape))

            reshape_cloak = cur_protected - cur_original

            reshape_cloak = reshape_cloak[0:org_shape[0], 0:org_shape[1], :]

            callback_id = self.callback_idx[i]####一个标号self
            bb = self.cropped_index[i]
            self.cloaked_faces[callback_id][bb[1]:bb[3], bb[0]:bb[2], :] += reshape_cloak

        for i in range(0, len(self.cloaked_faces)):
            self.cloaked_faces[i] = np.clip(self.cloaked_faces[i], 0.0, 255.0)
        return self.cloaked_faces


def dump_dictionary_as_json(dict, outfile):
    j = json.dumps(dict)
    with open(outfile, "wb") as f:
        f.write(j.encode())


def load_victim_model(number_classes, teacher_model=None, end2end=False):
    for l in teacher_model.layers:
        l.trainable = end2end
    x = teacher_model.layers[-1].output

    x = Dense(number_classes)(x)
    x = Activation('softmax', name="act")(x)
    model = Model(teacher_model.input, x)
    opt = keras.optimizers.Adadelta()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def resize(img, sz):####sz为一个元组img.shape = (1440,1440,3)
    assert np.min(img) >= 0 and np.max(img) <= 255.0###像素值的范围
    from keras.preprocessing import image
    im_data = image.array_to_img(img).resize((sz[1], sz[0]))###运用image的函数将图片重新定位为244*244
    im_data = image.img_to_array(im_data)###返回图片对应的的数组数据
    return im_data


def init_gpu(gpu_index, force=False):
    if isinstance(gpu_index, list):
        gpu_num = ','.join([str(i) for i in gpu_index])
    else:
        gpu_num = str(gpu_index)
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] and not force:
        print('GPU already initiated')
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    sess = fix_gpu_memory()
    return sess


def fix_gpu_memory(mem_fraction=0.8):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf_config = None
    if tf.test.is_gpu_available():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_fraction)
        tf_config = tf.ConfigProto(gpu_options=gpu_options)
        tf_config.gpu_options.allow_growth = True
        tf_config.log_device_placement = False
        
    init_op = tf.global_variables_initializer()  
    
    sess = tf.Session(config=tf_config)
    sess.run(init_op)
    K.set_session(sess)
    return sess


def preprocess(X, method):
    assert method in {'raw', 'imagenet', 'inception', 'mnist'}

    if method == 'raw':
        pass
    elif method == 'imagenet':
        X = imagenet_preprocessing(X)
    else:
        raise Exception('unknown method %s' % method)

    return X


def reverse_preprocess(X, method):
    assert method in {'raw', 'imagenet', 'inception', 'mnist'}

    if method == 'raw':
        pass
    elif method == 'imagenet':
        X = imagenet_reverse_preprocessing(X)
    else:
        raise Exception('unknown method %s' % method)

    return X


def imagenet_preprocessing(x, data_format=None):
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in ('channels_last', 'channels_first')

    x = np.array(x)
    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]

    mean = [103.939, 116.779, 123.68]
    std = None

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
        else:
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            if std is not None:
                x[:, 0, :, :] /= std[0]
                x[:, 1, :, :] /= std[1]
                x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]

    return x


def imagenet_reverse_preprocessing(x, data_format=None):
    import keras.backend as K
    x = np.array(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in ('channels_last', 'channels_first')

    if data_format == 'channels_first':
        if x.ndim == 3:
            # Zero-center by mean pixel
            x[0, :, :] += 103.939
            x[1, :, :] += 116.779
            x[2, :, :] += 123.68
            # 'BGR'->'RGB'
            x = x[::-1, :, :]
        else:
            x[:, 0, :, :] += 103.939
            x[:, 1, :, :] += 116.779
            x[:, 2, :, :] += 123.68
            x = x[:, ::-1, :, :]
    else:
        # Zero-center by mean pixel
        x[..., 0] += 103.939###np.min(x[:,:,:,0]) = -103.939
        x[..., 1] += 116.779
        x[..., 2] += 123.68
        # 'BGR'->'RGB'
        x = x[..., ::-1]
    return x


def reverse_process_cloaked(x, preprocess='imagenet'):
    # x = clip_img(x, preprocess)
    return reverse_preprocess(x, preprocess)


def build_bottleneck_model(model, cut_off):
    bottleneck_model = Model(model.input, model.get_layer(cut_off).output)
    bottleneck_model.compile(loss='categorical_crossentropy',
                             optimizer='adam',
                             metrics=['accuracy'])
    return bottleneck_model


def load_extractor(name):
    model_dir = os.path.join(os.path.expanduser('~'), 'fawkes')
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, "{}.h5".format(name))
    emb_file = os.path.join(model_dir, "{}_emb.p.gz".format(name))
    if os.path.exists(model_file):
        model = keras.models.load_model(model_file)
    else:
        print("Download models...")
        get_file("{}.h5".format(name), "http://mirror.cs.uchicago.edu/fawkes/files/{}.h5".format(name),
                 cache_dir=model_dir, cache_subdir='')
        model = keras.models.load_model(model_file)

    if not os.path.exists(emb_file):
        get_file("{}_emb.p.gz".format(name), "http://mirror.cs.uchicago.edu/fawkes/files/{}_emb.p.gz".format(name),
                 cache_dir=model_dir, cache_subdir='')

    if hasattr(model.layers[-1], "activation") and model.layers[-1].activation == "softmax":
        raise Exception(
            "Given extractor's last layer is softmax, need to remove the top layers to make it into a feature extractor")
    return model


def get_dataset_path(dataset):
    model_dir = os.path.join(os.path.expanduser('~'), '.fawkes')
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        raise Exception("Please config the datasets before running protection code. See more in README and config.py.")

    config = json.load(open(os.path.join(model_dir, "config.json"), 'r'))
    if dataset not in config:
        raise Exception(
            "Dataset {} does not exist, please download to data/ and add the path to this function... Abort".format(
                dataset))
    return config[dataset]['train_dir'], config[dataset]['test_dir'], config[dataset]['num_classes'], config[dataset][
        'num_images']


def dump_image(x, filename, format="png", scale=False):
    img = image.array_to_img(x, scale=scale)
    img.save(filename, format)
    return


def load_embeddings(feature_extractors_names):
    model_dir = os.path.join(os.path.expanduser('~'), '.fawkes')
    for extractor_name in feature_extractors_names:
        fp = gzip.open(os.path.join(model_dir, "{}_emb.p.gz".format(extractor_name)), 'rb')
        path2emb = pickle.load(fp)
        fp.close()

    return path2emb


def extractor_ls_predict(feature_extractors_ls, X):
    feature_ls = []
    for extractor in feature_extractors_ls:
        cur_features = extractor.predict(X)###X.shape = (1,224,224,3)
        feature_ls.append(cur_features)####cur_features.shape = (1,1024)
    concated_feature_ls = np.concatenate(feature_ls, axis=1)
    return concated_feature_ls



###B.shape = (18947,1024)
def pairwise_l2_distance(A, B):
    BT = B.transpose()
    vecProd = np.dot(A, BT)
    SqA = A ** 2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))

    SqB = B ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    return ED




def select_target_label_CLQ(imgs, feature_extractors_ls, feature_extractors_names, separate_target,metric='l2',index = 0,power = 20):
    model_dir = os.path.join(os.path.expanduser('~'), '.fawkes')

    original_feature_x = extractor_ls_predict(feature_extractors_ls, imgs)####1024个向量中，有467个向量都为零；

    path2emb = load_embeddings(feature_extractors_names)###dict

    items = list([(k, v) for k, v in path2emb.items()])
    ####len(items) = 18947    
    ###itmes[i] = (i,x),其中i是的i个target的标号，x则是1024维的向量，已经提取好的特征向量
    
    paths = [p[0] for p in items]###所有target的标号
    embs = [p[1] for p in items]###所有target向量提取的的特征向量可以发现，大概有423个均为0
    embs = np.array(embs)

    pair_dist = pairwise_l2_distance(original_feature_x, embs)####embs.shape = (18947,1024)十张图片所提取的特征向量，的中心或者说平均值来进行计算的
    
    ###这里我只放入了一张图片，得到了这张图片和18947张target图片的距离
    pair_dist = np.array(pair_dist)###ma
    
    
    ###最大化的最小值计算
    max_sum = np.min(pair_dist, axis=0)###18947维 ###得到了所有的最小值,所有的最小值为26.865223
    
    
    max_id_ls = np.argsort(max_sum)[::-1]###这个返回的结果标号表示从大到小

    #max_id = random.choice(max_id_ls[:20])###从前20个随机选择一个
    if separate_target:
        
        max_id = max_id_ls[index]
    
        target_data_id = paths[int(max_id)]###这里才是真正的target_id的标号
        print("target ID: {}".format(target_data_id))
        
        ###如果所选的target不存在,那么就要进行在线下载
    
        image_dir = os.path.join(model_dir, "target_data/{}".format(target_data_id))
    
        os.makedirs(os.path.join(model_dir, "target_data"), exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        for i in range(10):
            if os.path.exists(os.path.join(model_dir, "target_data/{}/{}.jpg".format(target_data_id, i))):
                continue
            try:
                get_file("{}.jpg".format(i),
                         "http://mirror.cs.uchicago.edu/fawkes/files/target_data/{}/{}.jpg".format(target_data_id, i),
                         cache_dir=model_dir, cache_subdir='target_data/{}/'.format(target_data_id))
            except Exception:
                pass
    
        image_paths = glob.glob(image_dir + "/*.jpg")
    
        ###target里面图片的像素是112*112*3
        target_images = [image.img_to_array(image.load_img(cur_path)) for cur_path in
                         image_paths]
    
        target_images = np.array([resize(x, (224, 224)) for x in target_images])###转换为224*224*3
        target_images = preprocess(target_images, 'imagenet')
    
        target_images = list(target_images)
        ###len(target_images) = 10
        while len(target_images) < len(imgs):
            target_images += target_images
    
        target_images = random.sample(target_images, len(imgs))
        return np.array(target_images)
    else:
        max_id = max_id_ls[:power]
        target_data_id_lt = [17826,7143,17829,20827,1544,
                  7478,6693,1681,19534,5095,
                  2882,8605,6997,12183,9297,
                  7954,18249,19341,18386,16888]
        target_images_list = []
        
        for i in range(power):
        
            
            #target_data_id = paths[int(max_id[i])]###这里才是真正的target_id的标号
            target_data_id = target_data_id_lt[i]
            
            print("target ID: {}".format(target_data_id))
            
            ###如果所选的target不存在,那么就要进行在线下载
        
            image_dir = os.path.join(model_dir, "target_data/{}".format(target_data_id))
        
            os.makedirs(os.path.join(model_dir, "target_data"), exist_ok=True)
            os.makedirs(image_dir, exist_ok=True)
            for i in range(10):
                if os.path.exists(os.path.join(model_dir, "target_data/{}/{}.jpg".format(target_data_id, i))):
                    continue
                try:
                    get_file("{}.jpg".format(i),
                             "http://mirror.cs.uchicago.edu/fawkes/files/target_data/{}/{}.jpg".format(target_data_id, i),
                             cache_dir=model_dir, cache_subdir='target_data/{}/'.format(target_data_id))
                except Exception:
                    pass
        
            image_paths = glob.glob(image_dir + "/*.jpg")
        
            ###target里面图片的像素是112*112*3
            target_images = [image.img_to_array(image.load_img(cur_path)) for cur_path in
                             image_paths]
        
            target_images = np.array([resize(x, (224, 224)) for x in target_images])###转换为224*224*3
            target_images = preprocess(target_images, 'imagenet')
        
            target_images = list(target_images)
            target_images_list.extend(random.sample(target_images, 1))
            ###len(target_images) = 10
        
        
        while len(target_images_list) < len(imgs):
            target_images_list.extend(target_images_list)
        
        
        
        target_images = target_images_list[0:len(imgs)]###这里面的target属于同一各种类的target ID
        
        
        
        return np.array(target_images)


def select_target_label(imgs, feature_extractors_ls, feature_extractors_names, metric='l2'):
    model_dir = os.path.join(os.path.expanduser('~'), '.fawkes')

    original_feature_x = extractor_ls_predict(feature_extractors_ls, imgs)####1024个向量中，有467个向量都为零；img.shape = (1,224,224,3)

    path2emb = load_embeddings(feature_extractors_names)###dict

    items = list([(k, v) for k, v in path2emb.items()])
    ####len(items) = 18947    
    ###itmes[i] = (i,x),其中i是的i个target的标号，x则是1024维的向量，已经提取好的特征向量
    
    paths = [p[0] for p in items]###所有target的标号
    embs = [p[1] for p in items]###所有target向量提取的的特征向量可以发现，大概有423个均为0
    embs = np.array(embs)
    
    ##pair_dist.shape = (1,18947)
    pair_dist = pairwise_l2_distance(original_feature_x, embs)####embs.shape = (18947,1024)十张图片所提取的特征向量，的中心或者说平均值来进行计算的
    
    ###这里我只放入了一张图片，得到了这张图片和18947张target图片的距离
    pair_dist = np.array(pair_dist)###ma
    
    
    ###最大化的最小值计算
    max_sum = np.min(pair_dist, axis=0)###18947维 ###得到了所有的最小值,所有的最小值为26.865223
    
    
    max_id_ls = np.argsort(max_sum)[::-1]###这个返回的结果标号表示从大到小

    #max_id = max_id_ls[0]
    max_id = random.choice(max_id_ls[:20])###从前20个随机选择一个

    target_data_id = paths[int(max_id)]###这里才是真正的target_id的标号
    print("target ID: {}".format(target_data_id))
    
    ###如果所选的target不存在,那么就要进行在线下载

    image_dir = os.path.join(model_dir, "target_data/{}".format(target_data_id))

    os.makedirs(os.path.join(model_dir, "target_data"), exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    for i in range(10):
        if os.path.exists(os.path.join(model_dir, "target_data/{}/{}.jpg".format(target_data_id, i))):
            continue
        try:
            get_file("{}.jpg".format(i),
                     "http://mirror.cs.uchicago.edu/fawkes/files/target_data/{}/{}.jpg".format(target_data_id, i),
                     cache_dir=model_dir, cache_subdir='target_data/{}/'.format(target_data_id))
        except Exception:
            pass

    image_paths = glob.glob(image_dir + "/*.jpg")

    ###target里面图片的像素是112*112*3
    target_images = [image.img_to_array(image.load_img(cur_path)) for cur_path in
                     image_paths]

    target_images = np.array([resize(x, (224, 224)) for x in target_images])###转换为224*224*3
    target_images = preprocess(target_images, 'imagenet')

    target_images = list(target_images)
    ###len(target_images) = 10
    while len(target_images) < len(imgs):
        target_images += target_images

    target_images = random.sample(target_images, len(imgs))
    return np.array(target_images)


""" TensorFlow implementation get_file
https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/utils/data_utils.py#L168-L297
"""


def get_file(fname,
             origin,
             untar=False,
             md5_hash=None,
             file_hash=None,
             cache_subdir='datasets',
             hash_algorithm='auto',
             extract=False,
             archive_format='auto',
             cache_dir=None):
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.fawkes')
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = 'md5'
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.fawkes')
    datadir = os.path.join(datadir_base, cache_subdir)
    _makedirs_exist_ok(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    download = False
    if not os.path.exists(fpath):
        download = True

    if download:
        error_msg = 'URL fetch failure on {}: {} -- {}'
        dl_progress = None
        try:
            try:
                urlretrieve(origin, fpath, dl_progress)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
        # ProgressTracker.progbar = None

    if untar:
        if not os.path.exists(untar_fpath):
            _extract_archive(fpath, datadir, archive_format='tar')
        return untar_fpath

    if extract:
        _extract_archive(fpath, datadir, archive_format)

    return fpath


def _extract_archive(file_path, path='.', archive_format='auto'):
    if archive_format is None:
        return False
    if archive_format == 'auto':
        archive_format = ['tar', 'zip']
    if isinstance(archive_format, six.string_types):
        archive_format = [archive_format]

    for archive_type in archive_format:
        if archive_type == 'tar':
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        if archive_type == 'zip':
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile

        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                try:
                    archive.extractall(path)
                except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                    if os.path.exists(path):
                        if os.path.isfile(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    raise
            return True
    return False


def _makedirs_exist_ok(datadir):
    if six.PY2:
        # Python 2 doesn't have the exist_ok arg, so we try-except here.
        try:
            os.makedirs(datadir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    else:
        os.makedirs(datadir, exist_ok=True)  # pylint: disable=unexpected-keyword-arg
