from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import align.detect_face
import random

from os.path import join as pjoin
import matplotlib.pyplot as plt


import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics  
from sklearn.externals import joblib





#face detection parameters
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold 三步的阈值
factor = 0.709 # scale factor 比例因子


# 模型位置
model_dir='./20170512-110547'#"Directory containing the graph definition and checkpoint files.")


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def read_img(person_dir,f):
    img=cv2.imread(pjoin(person_dir, f))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      
    # 判断数组维度
    if gray.ndim == 2:
        img = to_rgb(gray)
    return img

def load_data(data_dir):
    data = {}
    pics_ctr = 0
    for guy in os.listdir(data_dir):
        person_dir = pjoin(data_dir, guy)       
        curr_pics = [read_img(person_dir, f) for f in os.listdir(person_dir)]         
        # 存储每一类人的文件夹内所有图片
        data[guy] = curr_pics      
    return data

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

# 创建mtcnn网络，并加载参数
print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

def load_and_align_data(image, image_size, margin, gpu_memory_fraction):

    # 读取图片 
    img = image
    # 获取图片的shape
    img_size = np.asarray(img.shape)[0:2]
    # 返回边界框数组 （参数分别是输入图片 脸部最小尺寸 三个网络 阈值 factor不清楚）
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
  
    if len(bounding_boxes) < 1:
        return 0,0,0
    else:    
        crop=[]
        det=bounding_boxes

        det[:,0]=np.maximum(det[:,0], 0)
        det[:,1]=np.maximum(det[:,1], 0)
        det[:,2]=np.minimum(det[:,2], img_size[1])
        det[:,3]=np.minimum(det[:,3], img_size[0])

        # det[:,0]=np.maximum(det[:,0]-margin/2, 0)
        # det[:,1]=np.maximum(det[:,1]-margin/2, 0)
        # det[:,2]=np.minimum(det[:,2]+margin/2, img_size[1])
        # det[:,3]=np.minimum(det[:,3]+margin/2, img_size[0])

        det=det.astype(int)

        for i in range(len(bounding_boxes)):
            temp_crop=img[det[i,1]:det[i,3],det[i,0]:det[i,2],:]
            aligned=misc.imresize(temp_crop, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            crop.append(prewhitened)
        crop_image=np.stack(crop)
            
        return det,crop_image,1


with tf.Graph().as_default():
    with tf.Session() as sess:  
        # 加载模型
        facenet.load_model(model_dir)

        print('建立facenet embedding模型')
        # 返回给定名称的tensor
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    
        model = joblib.load('./models/knn_classifier.model')

        
        #开启ip摄像头
        video="http://admin:admin@192.168.0.107:8081/"   #此处@后的ipv4 地址需要修改为自己的地址
        # 参数为0表示打开内置摄像头，参数是视频文件路径则打开视频
        capture =cv2.VideoCapture(video)
        cv2.namedWindow("camera",1)
        c=0
        num = 0
        frame_interval=3 # frame intervals  
        while True:
            ret, frame = capture.read()
            timeF = frame_interval

            # print(shape(frame))
            detect_face=[]

            if(c%timeF == 0):
                find_results=[]
                # cv2.imshow("camera",frame)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if gray.ndim == 2:
                    img = to_rgb(gray)
                det,crop_image,j= load_and_align_data(img, 160, 44, 1.0)
                if j:
                    feed_dict = { images_placeholder: crop_image, phase_train_placeholder:False }        
                    emb = sess.run(embeddings, feed_dict=feed_dict) 

                    for xx in range(len(emb)):
                        print(type(emb[xx,:]),emb[xx,:].shape)
                        detect_face.append(emb[xx,:])
                    detect_face=np.array(detect_face)
                    detect_face=detect_face.reshape(-1,128)
                    print('facenet embedding模型建立完毕')

                    predict = model.predict(detect_face) 
                    print(predict)
                    result=[]

                    for i in range(len(predict)):
                        if predict[i]==0:
                            result.append('windzu')
                        elif predict[i]==100:
                            result.append('unknown')

                    # 绘制矩形框并标注文字
                    for rec_position in range(len(det)):
                        
                        cv2.rectangle(frame,(det[rec_position,0],det[rec_position,1]),(det[rec_position,2],det[rec_position,3]),(0, 255, 0), 2, 8, 0)

                        cv2.putText(
                            frame,
                        result[rec_position], 
                        (det[rec_position,0],det[rec_position,1]),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                        0.8, 
                        (0, 0 ,255), 
                        thickness = 2, 
                        lineType = 2)
                cv2.imshow('camera',frame)
        
            c+=1

            key = cv2.waitKey(3)

            if key == 27:
                #esc键退出
                print("esc break...")
                break

            if key == ord(' '):
                # 保存一张图像
                num = num+1
                filename = "frames_%s.jpg" % num
                cv2.imwrite(filename,frame)
            
        # When everything is done, release the capture
        capture.release()
        cv2.destroyWindow("camera")