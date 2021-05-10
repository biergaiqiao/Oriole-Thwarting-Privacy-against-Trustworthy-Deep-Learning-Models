#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-10-21
# @Author  : Emily Wenger (ewenger@uchicago.edu)

import time

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras.utils import Progbar


class FawkesMaskGeneration:
    # if the attack is trying to mimic a target image or a neuron vector
    MIMIC_IMG = True
    # number of iterations to perform gradient descent
    MAX_ITERATIONS = 10000
    # larger values converge faster to less accurate results
    LEARNING_RATE = 1e-2
    # the initial constant c to pick as a first guess
    INITIAL_CONST = 1
    # pixel intensity range
    INTENSITY_RANGE = 'imagenet'
    # threshold for distance
    L_THRESHOLD = 0.03
    # whether keep the final result or the best result
    KEEP_FINAL = False
    # max_val of image
    MAX_VAL = 255
    MAXIMIZE = False
    IMAGE_SHAPE = (224, 224, 3)
    RATIO = 1.0
    LIMIT_DIST = False
    LOSS_TYPE = 'features'  # use features (original Fawkes) or gradients (Witches Brew) to run Fawkes?

    def __init__(self, bottleneck_model_ls, mimic_img=MIMIC_IMG,
                 batch_size=1, learning_rate=LEARNING_RATE,
                 max_iterations=MAX_ITERATIONS, initial_const=INITIAL_CONST,
                 intensity_range=INTENSITY_RANGE, l_threshold=L_THRESHOLD,
                 max_val=MAX_VAL, keep_final=KEEP_FINAL, maximize=MAXIMIZE, image_shape=IMAGE_SHAPE, verbose=1,
                 ratio=RATIO, limit_dist=LIMIT_DIST, loss_method=LOSS_TYPE):

        assert intensity_range in {'raw', 'imagenet', 'inception', 'mnist'}

        # constant used for tanh transformation to avoid corner cases
        self.it = 0
        self.tanh_constant = 2 - 1e-6
        self.MIMIC_IMG = mimic_img
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.intensity_range = intensity_range
        self.l_threshold = l_threshold
        self.max_val = max_val
        self.keep_final = keep_final
        self.verbose = verbose
        self.maximize = maximize
        self.learning_rate = learning_rate
        self.ratio = ratio
        self.limit_dist = limit_dist
        self.single_shape = list(image_shape)
        self.bottleneck_models = bottleneck_model_ls
        self.loss_method = loss_method

        self.input_shape = tuple([self.batch_size] + self.single_shape)

        self.bottleneck_shape = tuple([self.batch_size] + self.single_shape)

        # the variable we're going to optimize over
        self.modifier = tf.Variable(np.ones(self.input_shape, dtype=np.float32) * 1e-6)
        self.const = tf.Variable(np.ones(batch_size) * self.initial_const, dtype=np.float32)
        self.mask = tf.Variable(np.ones(batch_size), dtype=np.bool)

    @staticmethod
    def resize_tensor(input_tensor, model_input_shape):
        if input_tensor.shape[1:] == model_input_shape or model_input_shape[1] is None:
            return input_tensor
        resized_tensor = tf.image.resize(input_tensor, model_input_shape[:2])
        return resized_tensor

    def input_space_process(self, img):
        if self.intensity_range == 'imagenet':
            mean = np.repeat([[[[103.939, 116.779, 123.68]]]], self.batch_size, axis=0)
            raw_img = (img - mean)
        else:
            raw_img = img
        return raw_img

    def reverse_input_space_process(self, img):
        if self.intensity_range == 'imagenet':
            mean = np.repeat([[[[103.939, 116.779, 123.68]]]], self.batch_size, axis=0)
            raw_img = (img + mean)
        else:
            raw_img = img
        return raw_img

    def clipping(self, imgs):
        imgs = self.reverse_input_space_process(imgs)
        imgs = np.clip(imgs, 0, self.max_val)
        imgs = self.input_space_process(imgs)
        return imgs

    def calc_dissim(self, source_raw, source_mod_raw):
        return 0.0, 0.0, 0.0
        # msssim_split = tf.image.ssim(source_raw, source_mod_raw, max_val=255.0)
        # dist_raw = (1.0 - tf.stack(msssim_split)) / 2.0
        # dist = tf.maximum(dist_raw - self.l_threshold, 0.0)
        # # dist_raw_sum = tf.reduce_sum(tf.where(self.mask, dist_raw, tf.zeros_like(dist_raw)))
        # dist_raw_sum = tf.reduce_sum(dist_raw)
        # # dist_sum = tf.reduce_sum(tf.where(self.mask, dist, tf.zeros_like(dist)))
        # dist_sum = tf.reduce_sum(dist)
        # return dist, dist_sum, dist_raw_sum

    def calc_bottlesim(self, tape, source_raw, target_raw, source_filtered, original_raw):
        """ original Fawkes loss function. """
        bottlesim = 0.0
        bottlesim_sum = 0.0
        # make sure everything is the right size.
        model_input_shape = self.single_shape
        cur_aimg_input = self.resize_tensor(source_raw, model_input_shape)
        cur_source_filtered = self.resize_tensor(source_filtered, model_input_shape)
        # cur_timg_input = self.resize_tensor(target_raw, model_input_shape)
        for bottleneck_model in self.bottleneck_models:
            if tape is not None:
                try:
                    tape.watch(bottleneck_model.variables)
                except AttributeError:
                    tape.watch(bottleneck_model.model.variables)
            # get the respective feature space reprs.
            bottleneck_a = bottleneck_model(cur_aimg_input)
            bottleneck_filter = bottleneck_model(cur_source_filtered)

            bottleneck_s = bottleneck_model(original_raw)

            # compute the differences.
            bottleneck_diff = bottleneck_a - bottleneck_s
            bottleneck_diff_filter = bottleneck_filter - bottleneck_s
            # get scale factor.
            scale_factor = tf.sqrt(tf.reduce_sum(tf.square(bottleneck_s), axis=1))
            scale_factor_filter = tf.sqrt(tf.reduce_sum(tf.square(bottleneck_diff_filter), axis=1))
            # compute the loss
            cur_bottlesim = tf.reduce_sum(tf.square(bottleneck_diff), axis=1)
            cur_bottlesim_filter = tf.reduce_sum(tf.square(bottleneck_diff_filter), axis=1)

            cur_bottlesim = cur_bottlesim / scale_factor
            cur_bottlesim_filter = cur_bottlesim_filter / scale_factor_filter

            bottlesim += cur_bottlesim + cur_bottlesim_filter
            bottlesim_sum += tf.reduce_sum(cur_bottlesim) + tf.reduce_sum(cur_bottlesim_filter)
        return bottlesim, bottlesim_sum

    def compute_feature_loss(self, tape, aimg_raw, simg_raw, aimg_input, timg_input, simg_input, aimg_filtered):
        """ Compute input space + feature space loss.
        """
        input_space_loss, input_space_loss_sum, input_space_loss_raw_sum = self.calc_dissim(aimg_raw, simg_raw)
        feature_space_loss, feature_space_loss_sum = self.calc_bottlesim(tape, aimg_input, timg_input, aimg_filtered, simg_input)

        if self.maximize:
            loss = self.const * input_space_loss - feature_space_loss
        else:
            if self.it < self.MAX_ITERATIONS:
                loss = self.const * input_space_loss + 1000 * feature_space_loss  # - feature_space_loss_orig
            else:
                loss = self.const * 100 * input_space_loss + feature_space_loss
        # loss_sum = tf.reduce_sum(tf.where(self.mask, loss, tf.zeros_like(loss)))
        loss_sum = tf.reduce_sum(loss)
        # return loss_sum, input_space_loss, feature_space_loss, input_space_loss_sum, input_space_loss_raw_sum, feature_space_loss_sum
        return loss_sum, 0, feature_space_loss, 0, 0, feature_space_loss_sum

    def attack(self, source_imgs, target_imgs, weights=None):
        """ Main function that runs cloak generation. """
        if weights is None:
            weights = np.ones([source_imgs.shape[0]] +
                              list(self.bottleneck_shape[1:]))

        assert weights.shape[1:] == self.bottleneck_shape[1:]
        assert source_imgs.shape[1:] == self.input_shape[1:]
        assert source_imgs.shape[0] == weights.shape[0]
        if self.MIMIC_IMG:
            assert target_imgs.shape[1:] == self.input_shape[1:]
            assert source_imgs.shape[0] == target_imgs.shape[0]
        else:
            assert target_imgs.shape[1:] == self.bottleneck_shape[1:]
            assert source_imgs.shape[0] == target_imgs.shape[0]

        start_time = time.time()

        adv_imgs = []
        print('%d batches in total'
              % int(np.ceil(len(source_imgs) / self.batch_size)))

        for idx in range(0, len(source_imgs), self.batch_size):
            # print('processing image %d at %s' % (idx + 1, datetime.datetime.now()))
            adv_img = self.attack_batch(source_imgs[idx:idx + self.batch_size],
                                        target_imgs[idx:idx + self.batch_size])
            adv_imgs.extend(adv_img)

        elapsed_time = time.time() - start_time
        print('protection cost %f s' % elapsed_time)

        return np.array(adv_imgs)

    def attack_batch(self, source_imgs, target_imgs):
        """ TF2 method to generate the cloak. """
        # preprocess images.
        global progressbar
        nb_imgs = source_imgs.shape[0]
        mask = [True] * nb_imgs + [False] * (self.batch_size - nb_imgs)
        self.mask = np.array(mask, dtype=np.bool)
        LR = self.learning_rate

        # make sure source/target images are an array
        source_imgs = np.array(source_imgs, dtype=np.float32)
        target_imgs = np.array(target_imgs, dtype=np.float32)

        # metrics to test
        best_bottlesim = [0] * nb_imgs if self.maximize else [np.inf] * nb_imgs
        best_adv = np.zeros(source_imgs.shape)
        total_distance = [0] * nb_imgs
        finished_idx = set()

        # make the optimizer
        optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        # optimizer = tf.keras.optimizers.Adadelta(self.learning_rate)

        # get the modifier
        self.modifier = tf.Variable(np.ones(self.input_shape, dtype=np.float32) * 1e-4)
        # self.modifier = tf.Variable(np.random.uniform(-8.0, 8.0, self.input_shape), dtype=tf.float32)

        if self.verbose == 0:
            progressbar = Progbar(
                self.MAX_ITERATIONS, width=30, verbose=1
            )

        # watch relevant variables.
        simg_tanh = tf.Variable(source_imgs, dtype=np.float32)
        timg_tanh = tf.Variable(target_imgs, dtype=np.float32)
        # simg_tanh = self.reverse_input_space_process(simg_tanh)
        # timg_tanh = self.reverse_input_space_process(timg_tanh)

        # run the attack
        self.it = 0
        below_thresh = False
        while self.it < self.MAX_ITERATIONS:
            self.it += 1
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(self.modifier)
                tape.watch(simg_tanh)
                tape.watch(timg_tanh)

                aimg_raw = simg_tanh + self.modifier
                aimg_filtered_raw = simg_tanh + tfa.image.gaussian_filter2d(self.modifier, [7, 7], 3.0)
                final_filtered_raw = simg_tanh + tfa.image.gaussian_filter2d(self.modifier, [1, 1], 2.0)

                simg_raw = simg_tanh
                timg_raw = timg_tanh

                # Convert further preprocess for bottleneck
                aimg_input = self.input_space_process(aimg_raw)
                aimg_filtered = self.input_space_process(aimg_filtered_raw)
                timg_input = self.input_space_process(timg_raw)
                simg_input = self.input_space_process(simg_raw)
                # aimg_input = aimg_raw
                # timg_input = timg_raw
                # simg_input = simg_raw
                # get the feature space loss.
                loss, input_dist, internal_dist, input_dist_sum, input_dist_raw_sum, internal_dist_sum = self.compute_feature_loss(
                    tape, aimg_raw, simg_raw, aimg_input, timg_input, simg_input, aimg_filtered)

                # compute gradients
                grad = tape.gradient(loss, [self.modifier])
                # grad[0] = grad[0] * 1e11
                grad[0] = tf.sign(grad[0]) * 0.6375
                # optimizer.apply_gradients(zip(grad, [self.modifier]))
                self.modifier = self.modifier - grad[0]
                self.modifier = tf.clip_by_value(self.modifier, -12.0, 12.0)

                for e, (feature_d, mod_img) in enumerate(zip(internal_dist, final_filtered_raw)):
                    if e >= nb_imgs:
                        break
                    if (feature_d < best_bottlesim[e] and (not self.maximize)) or (
                            feature_d > best_bottlesim[e] and self.maximize):
                        # print('found improvement')
                        best_bottlesim[e] = feature_d
                        best_adv[e] = mod_img
                # compute whether or not your perturbation is too big.
                # thresh_over = input_dist_sum / self.batch_size / self.l_threshold * 100
            # if self.it != 0 and (self.it % (self.MAX_ITERATIONS // 3) == 0):
            #     LR = LR * 0.8  # np.array([LR * 0.8])
            #     optimizer.learning_rate = LR
            #     print("LR: {}".format(LR))

            # print iteration result
            # if self.it % 10 == 0:
            if self.verbose == 1:
                thresh_over = input_dist_sum / self.batch_size / self.l_threshold * 100
                # import pdb
                # pdb.set_trace()
                print(
                    "ITER {:0.0f}  Total Loss: {:.4f} perturb: {:0.4f} ({:0.4f} over, {:0.4f} raw); sim: {:.4f}".format(
                        self.it, loss, input_dist_sum, thresh_over, input_dist_raw_sum,
                        internal_dist_sum / nb_imgs))

            if self.verbose == 0:
                progressbar.update(self.it)

        # DONE: print results
        if self.verbose == 1:
            thresh_over = input_dist_sum / self.batch_size / self.l_threshold * 100
            print(
                "END after {} iterations: Total Loss: {} perturb: {:0.4f} ({:0.4f} over, {:0.4f} raw); sim: {}".format(
                    self.it,
                    loss, input_dist_sum, thresh_over, input_dist_raw_sum, internal_dist_sum / nb_imgs))
        print("\n")
        best_adv = self.clipping(best_adv[:nb_imgs])

        return best_adv
