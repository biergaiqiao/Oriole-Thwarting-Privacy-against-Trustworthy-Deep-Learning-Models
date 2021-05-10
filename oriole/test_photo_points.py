# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 10:21:51 2020

@author: 一叶之秋
"""

###分析发现不同模式下的cloak的图片对应像素点改动是一致的，那么就可以表明，该

from keras.preprocessing import image
from fawkes.utils import filter_image_paths
import glob
import os
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import function


###我的loaded_images 0为原图，之后为0.001到0.01的DSSIM
###找到cloak过后的照片和原图相同的RGB对应的三维坐标

def image_process(loaded_images,excel_paths):
     
    for i in range(1,len(loaded_images)):
        
        index = np.argwhere(loaded_images[0] == loaded_images[i])
        print(len(index))
        df = pd.DataFrame(index)
        
        writer = pd.ExcelWriter(excel_paths,engine = 'openpyxl')
        book = load_workbook(excel_paths)
        writer.book = book
        writer.sheets = dict((ws.title,ws) for ws in book.worksheets)
        
        df.to_excel(writer,sheet_name = "result",startcol= 5 * (i - 1),index_label = '序号',header = ['X','Y','RGB'])
        writer.save()
        
            
    return

class images_procession:

    def image_recovery(img):
        
        
        return

def conclude(img_ori,img_clo):
    """
        验证cloak过程中的RGB同增同减趋势
        
    """
    
    
    """
    sum_total = 0
    sum_RGB = 0
    
    compare_R = img_ori[:,:,0] < img_clo[:,:,0]
    compare_G = img_ori[:,:,1] < img_clo[:,:,1]
    compare_B = img_ori[:,:,2] < img_clo[:,:,2]
    
    sum_total += compare_R.sum() + compare_G.sum() +compare_B.sum()
    
    print('<',compare_R.sum(),compare_G.sum(),compare_B.sum())

    compare_R = img_ori[:,:,0] == img_clo[:,:,0]
    compare_G = img_ori[:,:,1] == img_clo[:,:,1]
    compare_B = img_ori[:,:,2] == img_clo[:,:,2]
    
    sum_total += compare_R.sum() + compare_G.sum() +compare_B.sum()
    
    print('=',compare_R.sum(),compare_G.sum(),compare_B.sum())
    
    
    compare_R = img_ori[:,:,0] > img_clo[:,:,0]
    compare_G = img_ori[:,:,1] > img_clo[:,:,1]
    compare_B = img_ori[:,:,2] > img_clo[:,:,2]
    
    sum_total += compare_R.sum() + compare_G.sum() +compare_B.sum()
    sum_RGB += compare_R.sum() + compare_G.sum() +compare_B.sum()
    
    print('>',compare_R.sum(),compare_G.sum(),compare_B.sum())
    print("number of points = {} equal rate = {:.6f}".format(sum_RGB , sum_RGB / sum_total))
    
    """
    
    sum_total = 150528
    sum_RGB = 0
    
    number = 20
    
    compare_R = abs(img_ori[:,:,0] - img_clo[:,:,0]) <= number
    compare_G = abs(img_ori[:,:,1] - img_clo[:,:,1]) <= number
    compare_B = abs(img_ori[:,:,2] - img_clo[:,:,2]) <= number
    
    sum_RGB += compare_R.sum() + compare_G.sum() +compare_B.sum()
    
    print("number of points = {} equal rate = {:.6f}".format(sum_RGB , sum_RGB / sum_total))
    
    
    return 
 


def PRINT(x,y,step,img):
    print('*'*150)
    for i in range(x,x+step):
        for j in range(y,y+step):
            print(img[i][j],end = ' ')
        print('\n')
    
     
    return 

def image_process_under_standard_size(loaded_images):
    from fawkes.utils import resize
    cur_faces_square = []    
    for i in range(len(loaded_images)):
        long_size = max([loaded_images[i].shape[1],loaded_images[i].shape[0]])
        base = np.zeros((long_size,long_size,3))
        base[0:loaded_images[i].shape[0],0:loaded_images[i].shape[1],:] = loaded_images[i]
        cur_faces_square.append(base)
    cur_faces_square = [resize(f,(224,224)) for f in cur_faces_square]
    #start_X = 100;start_Y = 100; step = 11
    
 
    
    
    """for i in range(1,11):
        print("\ncomparation between the original and the {}th image".format(i))
        conclude(cur_faces_square[0],cur_faces_square[i])
    for i in range(1,11):
        print("\nround {}th:".format(i))
        print("photo 0")
        PRINT(start_X,start_Y,step,cur_faces_square[0])
        print("photo {}".format(i))
        PRINT(start_X,start_Y,step,cur_faces_square[i])
        """
    
    #excel_path = r'F:\研二上\钱老板给的项目\结论\附件\compare_result_standard.xlsx'
   # image_process(cur_faces_square,excel_path)
    
    
    

        
    
    
    
    return cur_faces_square

def main():
     
    PATH = r'C:\Users\一叶之秋\Pictures\fawkes\total'
    path_excel = r'C:\Users\一叶之秋\Pictures\fawkes\compare_result.xlsx'
    
    image_paths = glob.glob(os.path.join(PATH,"*"))
    
    
    image_paths,loaded_images = filter_image_paths(image_paths)
    
    image_process_under_standard_size(loaded_images)
    
    """for i in range(1,11):
        print("\ncomparation between the original and the {}th image".format(i))
        conclude(loaded_images[0],loaded_images[i])"""
   # image_process(loaded_images,path_excel)
    
    return

if __name__ == '__main__':
    main()