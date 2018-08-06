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
  
    nrof_faces = bounding_boxes.shape[0]#number of faces
        #print('找到人脸数目为：{}'.format(nrof_faces))
    
    if nrof_faces==0:
        return 0,0,0
    else:    
        dataset={}
        position={}
        for face_position in bounding_boxes:      
            face_position=face_position.astype(int)    

            print((int(face_position[0]), int( face_position[1]),int(face_position[2]),int(face_position[3])))
            if face_position[0]>=0 and face_position[1]>=0:
            #word_position.append((int(face_position[0]), int( face_position[1])))
            
                cv2.rectangle(img, (face_position[0], 
                                face_position[1]), 
                            (face_position[2], face_position[3]), 
                            (0, 255, 0), 2)
                
                crop=img[face_position[1]:face_position[3],face_position[0]:face_position[2],]

                crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC )

                data=crop.reshape(-1,160,160,3)
                dataset=data
                position=face_position
            else:
                return 0,0,0

    return position,dataset,1

with tf.Graph().as_default():
    with tf.Session() as sess:  
        # 加载模型
        facenet.load_model(model_dir)

        print('建立facenet embedding模型')
        # 返回给定名称的tensor
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    
        model = joblib.load('./align/knn_classifier.model')

        
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

            if(c%timeF == 0):
                find_results=[]
                # cv2.imshow("camera",frame)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if gray.ndim == 2:
                    img = to_rgb(gray)

                # cv2.imshow('',img)

                position,images_me,j= load_and_align_data(img, 160, 44, 1.0)

                if j:
                    feed_dict = { images_placeholder: images_me, phase_train_placeholder:False }
                 
                    emb = sess.run(embeddings, feed_dict=feed_dict) 
                    print('facenet embedding模型建立完毕')

                    
                    emb1=emb.reshape(1,128)

                    print(type(emb1))
                    print(emb1.shape)

                    predict = model.predict(emb1) 

                    print(predict)
                    
                    if predict==1:
                        find_results.append('Handsome Zu')
                    elif predict==2:
                        find_results.append('others')
                
                    cv2.putText(frame,'detected:{}'.format(find_results), (50,100), 
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0 ,0), 
                            thickness = 2, lineType = 2)

                    cv2.rectangle(frame,(position[0],position[1]),(position[2],position[3]),(0, 255, 0), 2, 8, 0)#画矩形框

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