# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 11:11:19 2020

@author: 一叶之秋
"""
import argparse
import glob
import logging
import os
import sys

import tensorflow as tf

logging.getLogger('tensorflow').disabled = True

import numpy as np
from fawkes.differentiator import FawkesMaskGeneration
from fawkes.utils import load_extractor, init_gpu, select_target_label, dump_image, reverse_process_cloaked, \
    Faces, filter_image_paths,filter_image_paths_CLQ,resize,select_target_label_CLQ

import F_align_face
from fawkes.utils import get_file
from keras.preprocessing import image

import store_viariable
import function
import test_photo_points



class parser:
    def __init__(self,directory = 'imgs/',gpu = 0,mode = 'min',feature_extractor = 'high_extract',
                 th=0.01,max_step=1000,sd=1e9,lr=2,batch_size=1,separate_target=False,
                 no_align = False,debug=False,format='png',DEEP=1,power=20):
        self.directory = directory
        self.gpu = gpu
        self.mode = mode
        self.feature_extractor = feature_extractor
        self.th = th
        self.max_step = max_step
        self.sd = sd
        self.lr = lr
        self.batch_size = batch_size
        self.separate_target = separate_target  
        self.no_align = no_align
        self.debug = debug
        self.format = format
        self.DEEP = DEEP
        self.power = power

def generate_cloak_images(protector, image_X, target_emb=None):
    cloaked_image_X = protector.attack(image_X, target_emb)
    return cloaked_image_X


class Fawkes(object):
    
    #@JoeyChen prepare model and feature extractors
    
    def __init__(self, feature_extractor, gpu, batch_size):

        self.feature_extractor = feature_extractor
        self.gpu = gpu
        self.batch_size = batch_size
        global sess
        sess = init_gpu(gpu,force=True)
        global graph
        ##Clears the default graph stack and resets the global default graph.
        graph = tf.get_default_graph()
        
        
        
        model_dir = os.path.join(os.path.expanduser(store_viariable.PATH), '.fawkes')
        if not os.path.exists(os.path.join(model_dir, "mtcnn.p.gz")):
            os.makedirs(model_dir, exist_ok=True)##为True则目录存在的时候不弹出错误
            get_file("mtcnn.p.gz", "http://mirror.cs.uchicago.edu/fawkes/files/mtcnn.p.gz", cache_dir=model_dir,
                     cache_subdir='')

        self.fs_names = [feature_extractor]
        if isinstance(feature_extractor, list):
            self.fs_names = feature_extractor

        self.aligner = F_align_face.aligner(sess)
        self.feature_extractors_ls = [load_extractor(name) for name in self.fs_names]

        self.protector = None
        self.protector_param = None

    def mode2param(self, mode):
        if mode == 'min':
            th = 0.002
            max_step = 20
            lr = 40
        elif mode == 'low':
            th = 0.003
            max_step = 50
            lr = 35
        elif mode == 'mid':
            th = 0.005
            max_step = 200
            lr = 20
        elif mode == 'high':
            th = 0.008
            max_step = 500
            lr = 10
        elif mode == 'ultra':
            if not tf.test.is_gpu_available():
                print("Please enable GPU for ultra setting...")
                sys.exit(1)
            th = 0.01
            max_step = 1000
            lr = 8
        else:
            raise Exception("mode must be one of 'min', 'low', 'mid', 'high', 'ultra', 'custom'")
        return th, max_step, lr
    
    def run_protection_CLQ(self, image_paths, mode='min', th=0.04, sd=1e9, lr=10, max_step=500, batch_size=1, format='png',separate_target=True, debug=False, no_align=False,power = 20,DEEP = 1):
        
        """
        用于制作一张图片的前二十个cloaks版本
        """
        
        if mode == 'custom':
            pass
        else:
            th, max_step, lr = self.mode2param(mode)

        current_param = "-".join([str(x) for x in [mode, th, sd, lr, max_step, batch_size, format,
                                                   separate_target, debug]])

        image_paths, loaded_images = filter_image_paths_CLQ(image_paths,power = power,mode=mode,th = th)
        
        
        numbers = len(image_paths) // power
        
        cur_image_paths = []
        cur_loaded_images = []
        
        for i in range(numbers):
                   
            cur_image_paths.extend(image_paths[power * i :(i+1) * power])
            cur_loaded_images.extend(loaded_images[power * i:(i+1) * power])
           # image_paths, loaded_images = filter_image_paths(image_paths)
    
            if not cur_image_paths:
                print("No images in the directory")
                return 3
    
        with graph.as_default():
            faces = Faces(cur_image_paths, cur_loaded_images, self.aligner, verbose=1, no_align=no_align)
            original_images = faces.cropped_faces####original_images.shape = (1,224,224,3)

            if len(original_images) == 0:
                print("No face detected. ")
                return 2
            ###type(original_images) = 'numpy.ndarray'
            original_images = np.array(original_images)###original_images.shape = (1,224,224,3)

            with sess.as_default():
                if separate_target:
                    target_embedding = []
                    index = 0
                    for org_img in original_images:
                        org_img = org_img.reshape([1] + list(org_img.shape))
                        tar_emb = select_target_label_CLQ(org_img, self.feature_extractors_ls, self.fs_names,separate_target = separate_target,index = index % power,power = power)
                        target_embedding.append(tar_emb)
                        index = (index + 1) % power
                    target_embedding = np.concatenate(target_embedding)
                else:
                    target_embedding = select_target_label_CLQ(original_images, self.feature_extractors_ls, self.fs_names,separate_target = separate_target,index = 20,power = power)

                if current_param != self.protector_param:
                    self.protector_param = current_param

                    if self.protector is not None:
                        del self.protector

                    self.protector = FawkesMaskGeneration(sess, self.feature_extractors_ls,
                                                          batch_size=batch_size,
                                                          mimic_img=True,
                                                          intensity_range='imagenet',
                                                          initial_const=sd,
                                                          learning_rate=lr,
                                                          max_iterations=max_step,
                                                          l_threshold=th,
                                                          verbose=1 if debug else 0,
                                                          maximize=False,
                                                          keep_final=False,
                                                          image_shape=(224, 224, 3))

                protected_images = generate_cloak_images(self.protector, original_images,
                                                         target_emb=target_embedding)

                #original_images = original_images[0:self.power]
                #image_paths = image_paths[0:self.power]
                
                faces.cloaked_cropped_faces = protected_images

                final_images = faces.merge_faces(reverse_process_cloaked(protected_images),
                                                 reverse_process_cloaked(original_images))
            index = 0
            for i,(p_img, path) in enumerate(zip(final_images, cur_image_paths)):
                """long_size = max([len(loaded_images[i]),len(loaded_images[i][0])])
                im_data = image.array_to_img(p_img).resize((long_size, long_size))###运用image的函数将图片重新定位为244*244
                im_data = image.img_to_array(im_data)
                p_img = np.zeros((len(loaded_images[i]),len(loaded_images[i][0])))
                p_img = im_data[0:len(loaded_images[i]),0:len(loaded_images[i][0]),:]
                """
                
                path = ".".join(path.split(".")[:-1]) + "_{}_.jpg".format(index)
                
                file_name = "{}_{}_cloaked.{}".format(".".join(path.split(".")[:-1]), mode, format)
                dump_image(p_img, file_name, format=format)
                
                index = (index + 1) % 20
    
            print("Done!")
        return True

    def run_protection(self, image_paths, mode='min', th=0.04, sd=1e9, lr=10, max_step=500, batch_size=1, format='png',
                       separate_target=True, debug=False, no_align=False):
        if mode == 'custom':
            pass
        else:
            th, max_step, lr = self.mode2param(mode)

        current_param = "-".join([str(x) for x in [mode, th, sd, lr, max_step, batch_size, format,
                                                   separate_target, debug]])
        
        image_paths, loaded_images = filter_image_paths(image_paths)

        if not image_paths:
            print("No images in the directory")
            return 3

        with graph.as_default():
            faces = Faces(image_paths, loaded_images, self.aligner, verbose=1, no_align=no_align)
            original_images = faces.cropped_faces####original_images.shape = (1,224,224,3)

            if len(original_images) == 0:
                print("No face detected. ")
                return 2
            ###type(original_images) = 'numpy.ndarray'
            original_images = np.array(original_images)###original_images.shape = (1,224,224,3)
            
            ##计算特征向量
            #points = function.compute_points(original_images,self.feature_extractors_ls)
            
            
            
            with sess.as_default():
                if separate_target:
                    target_embedding = []
                    for org_img in original_images:
                        org_img = org_img.reshape([1] + list(org_img.shape))
                        tar_emb = select_target_label(org_img, self.feature_extractors_ls, self.fs_names)
                        target_embedding.append(tar_emb)
                    target_embedding = np.concatenate(target_embedding)
                else:
                    target_embedding = select_target_label(original_images, self.feature_extractors_ls, self.fs_names)

                if current_param != self.protector_param:
                    self.protector_param = current_param

                    if self.protector is not None:
                        del self.protector

                    self.protector = FawkesMaskGeneration(sess, self.feature_extractors_ls,
                                                          batch_size=batch_size,
                                                          mimic_img=True,
                                                          intensity_range='imagenet',
                                                          initial_const=sd,
                                                          learning_rate=lr,
                                                          max_iterations=max_step,
                                                          l_threshold=th,
                                                          verbose=1 if debug else 0,
                                                          maximize=False,
                                                          keep_final=False,
                                                          image_shape=(224, 224, 3))
                ###得到的protected_images里面的
                protected_images = generate_cloak_images(self.protector, original_images,
                                                         target_emb=target_embedding)
                

                faces.cloaked_cropped_faces = protected_images

                final_images = faces.merge_faces(reverse_process_cloaked(protected_images),
                                                 reverse_process_cloaked(original_images))

        for i,(p_img, path) in enumerate(zip(final_images, image_paths)):
    
            """long_size = max([len(loaded_images[i]),len(loaded_images[i][0])])
            im_data = image.array_to_img(p_img).resize((long_size, long_size))###运用image的函数将图片重新定位为244*244
            im_data = image.img_to_array(im_data)
            p_img = np.zeros((len(loaded_images[i]),len(loaded_images[i][0])))
            p_img = im_data[0:len(loaded_images[i]),0:len(loaded_images[i][0]),:]"""
            file_name = "{}_{}_cloaked.{}".format(".".join(path.split(".")[:-1]), mode, format)
            dump_image(p_img, file_name, format=format)

        print("Done!")
        
        
        return 1
    
def main(*argv): 
           
        
   

    
    if not argv:
        argv = list(sys.argv)
    
    try:
        import signal
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    except Exception as e:
        pass
    
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--power',help='cloak the most longest powerth target version',type=int,default=1)
    parser.add_argument('--DEEP', help = 'the depth of the folder',type = int,default=1)
    parser.add_argument('--directory', '-d', type=str,
                        help='the directory that contains images to run protection', default='imgs/')

    parser.add_argument('--gpu', '-g', type=str,
                        help='the GPU id when using GPU for optimization', default='0')

    parser.add_argument('--mode', '-m', type=str,
                        help='cloak generation mode, select from min, low, mid, high. The higher the mode is, the more perturbation added and stronger protection',
                        default='min')

    parser.add_argument('--feature-extractor', type=str,
                        help="name of the feature extractor used for optimization, currently only support high_extract",
                        default="high_extract")

    parser.add_argument('--th', help='only relevant with mode=custom, DSSIM threshold for perturbation', type=float,
                        default=0.002)
    parser.add_argument('--max-step', help='only relevant with mode=custom, number of steps for optimization', type=int,
                        default=1000)
    parser.add_argument('--sd', type=int, help='only relevant with mode=custom, penalty number, read more in the paper',
                        default=1e9)
    parser.add_argument('--lr', type=float, help='only relevant with mode=custom, learning rate', default=2)

    parser.add_argument('--batch-size', help="number of images to run optimization together", type=int, default=1)
    parser.add_argument('--separate_target', help="whether select separate targets for each faces in the directory",
                        action='store_true')
    parser.add_argument('--no-align', help="whether to detect and crop faces",
                        action='store_true')
    parser.add_argument('--debug', help="turn on debug and copy/paste the stdout when reporting an issue on github",
                        action='store_true')
    parser.add_argument('--format', type=str,
                        help="format of the output image",
                        default="png")

    argv = parser.parse_args(argv[1:])
    
   

    
    ##参数输入，原项目是采用从cmd进行输入的参数输入方式，这里我改成了用class实现参数输入
    
    #path = r'C:\Users\一叶之秋\Pictures\1\1'
    #path = r'D:\dataset\pubfig\fawkes_pub160\pub_final_usefull_split_10\PCA\Oriole\n000000'
    #argv = parser(path,gpu = 0,batch_size = 1,mode = 'min',power=1,DEEP = 1,th=0.002)
    
    assert argv.format in ['png', 'jpg', 'jpeg']    
    if argv.format == 'jpg':
        argv.format = 'jpeg'
    
    
    ###获取所有需要进行cloak的图片的文件（包含目录的文件）
    
    ###这段代码为手动确定cloaks的存储文件夹
    
    
    if argv.DEEP == 1:
        
        
        ##制作存储cloak的存储文件夹
        temp = '\\'.join(argv.directory.split('\\')[:-2])
        
        if argv.power == 1: 
            name = "{}_{}_{}{}_cloaks\\{}".format(argv.directory.split('\\')[-2],'fawkes',argv.mode,int(argv.th*1000),argv.directory.split('\\')[-1])
        else:
            name = '{}_{}_{}{}_cloaks\\{}'.format(argv.directory.split('\\')[-2],'oriole',argv.mode,int(argv.th*1000),argv.directory.split('\\')[-1])
            
        path_cloaked = os.path.join(temp,name)
        print(path_cloaked)
        if not os.path.exists(path_cloaked):
            os.makedirs(path_cloaked,exist_ok=True)
        
        image_paths_list = []
       
        image_paths= glob.glob(os.path.join(argv.directory, "*"))

        
        ###检查出没有cloaked的图片的含目录文件
        image_paths = [path for path in image_paths if "_cloaked" not in path.split("/")[-1]]
        
        
        
        protector = Fawkes(argv.feature_extractor, argv.gpu, argv.batch_size)
        ##pathO = r'C:\Users\一叶之秋\Pictures\fawkes\exam\custom0.01\origin_queen.jpg'
        ##pachC = r'C:\Users\一叶之秋\Pictures\fawkes\exam\custom0.01\origin_queen_custom_cloaked.jpeg'
        
        #@JoeyChen: 制作一种模式，还是四种模式都进行cloacked
        if argv.mode == 'all':
            for mode in ['min', 'low', 'mid', 'high']:
                protector.run_protection_CLQ(image_paths, mode=argv.mode, th=argv.th, sd=argv.sd, lr=argv.lr,
                                         max_step=argv.max_step,
                                         batch_size=argv.batch_size, format=argv.format,
                                         separate_target=argv.separate_target, debug=argv.debug, no_align=argv.no_align,power=argv.power,DEEP = argv.DEEP)
                    
        else:
                protector.run_protection_CLQ(image_paths, mode=argv.mode, th=argv.th, sd=argv.sd, lr=argv.lr,
                                     max_step=argv.max_step,
                                     batch_size=argv.batch_size, format=argv.format,
                                     separate_target=argv.separate_target, debug=argv.debug, no_align=argv.no_align,power = argv.power,DEEP = argv.DEEP,)
    else:
        
            ##制作存储cloak的存储文件夹
        temp = '\\'.join(argv.directory.split('\\')[:-1])
        
        if argv.power == 1: 
            name = "{}_{}_{}{}_cloaks".format(argv.directory.split('\\')[-1],argv.mode,int(argv.th*1000),'fawkes')
        else:
            name = '{}_{}_{}{}_cloaks'.format(argv.directory.split('\\')[-1],argv.mode,int(argv.th*1000),'oriole')
            
        path_cloaked = os.path.join(temp,name)
        
        if not os.path.exists(path_cloaked):
            os.makedirs(path_cloaked,exist_ok=True)
        
        root_image_paths =  glob.glob(os.path.join(argv.directory, "*"))
        
        image_paths = []
        
        for i in range(len(root_image_paths)):
            temp_dir = root_image_paths[i]
            temp_dir_lt = temp_dir.split('\\')
            temp_dir_lt[-2] = name
            temp_dir = '\\'.join(temp_dir_lt)
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir,exist_ok = True)
                
                
            image_paths_list = glob.glob(os.path.join(root_image_paths[i],"*"))
            image_paths_list = [path for path in image_paths_list if "_cloaked" not in path.split("/")[-1]]
            image_paths.extend(image_paths_list)
        
        protector = Fawkes(argv.feature_extractor, argv.gpu, argv.batch_size)
        ##pathO = r'C:\Users\一叶之秋\Pictures\fawkes\exam\custom0.01\origin_queen.jpg'
        ##pachC = r'C:\Users\一叶之秋\Pictures\fawkes\exam\custom0.01\origin_queen_custom_cloaked.jpeg'
        
        #@JoeyChen: 制作一种模式，还是四种模式都进行cloacked
        if argv.mode == 'all':
            for mode in ['min', 'low', 'mid', 'high']:
                protector.run_protection_CLQ(image_paths, mode=argv.mode, th=argv.th, sd=argv.sd, lr=argv.lr,
                                         max_step=argv.max_step,
                                         batch_size=argv.batch_size, format=argv.format,
                                         separate_target=argv.separate_target, debug=argv.debug, no_align=argv.no_align)
                                  
                
        else:
            
            protector.run_protection_CLQ(image_paths, mode=argv.mode, th=argv.th, sd=argv.sd, lr=argv.lr,
                                     max_step=argv.max_step,
                                     batch_size=argv.batch_size, format=argv.format,
                                     separate_target=argv.separate_target, debug=argv.debug, no_align=argv.no_align,DEEP=argv.DEEP,power = argv.power)
            #print("\nROUND {} FINISHED!\n".format(i+1))
        
        print("\nALL FINISHED!!!\n")
        
        

        


if __name__ == '__main__':
    main()
    

