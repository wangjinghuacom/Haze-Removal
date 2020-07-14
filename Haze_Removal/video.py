# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 19:45:49 2020

@author: 16332
"""
import cv2
import os
import glob
from guided_filter import dehaze

#将视频按帧切成图片
input_path = r'F:/Haze_Removal/save/1.mp4'
frame_path = r'F:/Haze_Removal/save/frames/'
def get_frame(input_path, frame_path):
    if not os.path.exists(frame_path):
        os.mkdir(frame_path)

    cap = cv2.VideoCapture(input_path)
    frame_count = 1
    success = True
    while(success):
        success, frame = cap.read()
        if success and frame_count%100 == 0:
            imshow('{}.img'.format(frame_count), frame)
        print('Read a frame: ', success)
        if success and frame_count%20 == 0:
            cv2.imwrite('save/frames/{}.jpg'.format(frame_count),frame,[100])
        frame_count += 1
        cv2.waitKey(1)
    cap.release()

def merge(img):
    size = (img.shape[1], img.shape[0])
    print(size)
#完成写入对象的创建，第一个参数是合成之后的视频的名称，
#第二个参数是可以使用的编码器，第三个参数是帧率即每秒钟展示多少张图片，
#第四个参数是图片大小信
    video = cv2.VideoWriter(r'F:/Haze_Removal/save/test.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 5, size)
    for f in glob.glob(r'F:/Haze_Removal/save/fra/*.jpg'):
        print(f)
        
        img = cv2.imread(f)
#        imshow('f',img)
        video.write(img)
    print('end')

def imshow(name, img):
    cv2.imshow(name,img)
    cv2.waitKey(3000)
#    cv2.imwrite('{}.jpg'.format(count),img,[100])
    cv2.destroyAllWindows()
# 取帧去雾处理保存
def get_result(frame_path):
        frames = os.listdir(frame_path)
        print(frames)
        print(len(frames))
        #for f in glob.glob(r'F:/Haze_Removal/save/frames/*.jpg'):
        for f in frames:
            print(f)
            path = os.path.join(frame_path + f)
            print(path)
            im = cv2.imread(path)
        #    imshow('fack', im)
            res = dehaze(im)
            cv2.imwrite('save/fra/{}.jpg'.format(f.split('.')[0]),res*255,[100])

path = r'F:/Haze_Removal/save/fra/20.jpg'
img = cv2.imread(path)
merge(img)


