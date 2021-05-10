# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 08:45:03 2020

@author: 一叶之秋
"""
import numpy as np
from fawkes.utils import extractor_ls_predict
from fawkes.utils import pairwise_l2_distance


####这里的loaded_images必须进行归一化处理
def compute_points(loaded_images,feature_extractors_ls):
    
    points = np.zeros((len(loaded_images),1024))
    points = extractor_ls_predict(feature_extractors_ls,loaded_images)###这是对所有加载的图片进行预测
    mean = np.average(points[1:],axis = 0)
    radius = pairwise_l2_distance(mean.reshape((1,1024)),points[1:,:])
    
    original_distance = pairwise_l2_distance(points[0,:],mean.reshape(1,1024))
    
    return points