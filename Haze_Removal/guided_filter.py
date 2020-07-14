#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 21:44:35 2020

@author: wjh
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def dark_channel(img, size=15):
    r,g,b = cv2.split(img)
    min_img = cv2.min(r, cv2.min(g,b))
    # 这个函数的第一个参数表示内核的形状，有三种形状可以选择。
    # 矩形：MORPH_RECT;交叉形：MORPH_CROSS;椭圆形：MORPH_ELLIPSE
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    # 获得size*size的锚框
    dc_img = cv2.erode(min_img, kernel)
    return dc_img

# 获取大气A值，具体做法是在暗通道中获取亮度为前0.1%的像素，
# 再在这些像素中寻找亮度最高的值作为A值
def get_A(img, percent=0.001):
    # 论文中的A是取原始像素中的某一个点的像素，这里取符合条件的所有点的平均值作为A的值，这样做因为
    # 如果取一个点，则各通道的A值很有可能全部很接近255，这样的话会造成处理后的图像偏色和出现大量色斑
    # 每个单元的每行平均值值组成一行一维数组
    mean_perpix = np.mean(img, axis = 2).reshape(-1)
    mean_topper = mean_perpix[:int(img.shape[0]*img.shape[1]*percent)]
    a = np.mean(mean_topper)
    return a if a<200 else 200 
# 每个通道的数据都需要除以对应的A值，即归一化，这样做，还存在一个问题，由于A的选取过程，
# 并不能保证每个像素分量值除以A值后都小于1，从而导致t的值可能小于0，而这是不容许的
def get_trans(img, a, w=0.95):
    x = img/a
    t = a - w*dark_channel(x, 15)
#    if t.all() < 0.1:
#        t = 0.1
    return t

def guided_filter(p, i, r, e):
    '''
    :param p: input image
    :param i: guidance image
    :param r: radius
    :param e: regularization
    :return: filtering output q
'''
    # corr为相关
    mean_I = cv2.boxFilter(i, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    corr_I = cv2.boxFilter(i * i, cv2.CV_64F, (r, r))
    corr_Ip = cv2.boxFilter(i * p, cv2.CV_64F, (r, r))
    # var为方差，cov为协方差
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    # a，b为线性系数
    a = cov_Ip / (var_I + e)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    
    q = mean_a * i + mean_b

    return q

def dehaze(im):
    # 导向滤波图计算需要在[0,1]范围内进行，也就是说导向图和预估的透射率图都必须
    # 从[0,255]先映射到[0,1]在进行计算
    img = im.astype('float64') / 255
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('float64') / 255
    
    a = get_A(img)
    t = get_trans(img, a)
    imshow('t_img', t) #保存时要*255，因为t值归一化了
    
    trans_guided = guided_filter(t, img_gray, 20, 0.0001)
    imshow('guide_filter', trans_guided) #保存时要*255
    trans_guided = cv2.max(trans_guided, 0.25)
    
    result = np.empty_like(img)
    for i in range(3):
        result[:,:,i] = (img[:,:,i] - a)/trans_guided + a
    
    imshow('result', result)
    
#    count = 1
#    cv2.imwrite('save/fra/{}.jpg'.format(count),result*255,[100])
#    count += 1
    return result
    

def imshow(name, img):
    cv2.imshow(name,img)
    cv2.waitKey(3000)
#    cv2.imwrite('{}3.jpg'.format(name),img,[100])
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    path = r'F:/Haze_Removal/3.png'
    img = cv2.imread(path)
    print(img.shape)
    imshow('source', img)
#    print(img[:,:])
#    #备份cap,直接读取第三列（下标从0开始）
#    cap=plt.imread(path)
#    print(cap.shape)
#    cap_alpha=cap.copy()
#    cap_alpha[:,:,3]=10
#    # 给第三列赋值 10 ，即增加最后一列的透明度
#    cap_alpha[:,:,3]=10
#    imshow('cap_alpha', cap_alpha)
    dehaze(img)
    
    
    dc_img = dark_channel(img)
    imshow('dack', dc_img)
    print(dc_img.shape)
#    print(dc_img[:,:])
    
    
    
    
    
    
    
    
    
    
    