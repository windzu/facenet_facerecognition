from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2

from os.path import join as pjoin
import matplotlib.pyplot as plt


import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics  
from sklearn.externals import joblib



from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import align.detect_face


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




# 加载图片并返回处理好的脸部图片 （参数列表  输入图片的路径 预期剪裁大小 边框大小 gpu选项）           
# def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

#     minsize = 20 # minimum size of face
#     threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
#     factor = 0.709 # scale factor
    
#     # 创建mtcnn网络，并加载参数
#     print('Creating networks and loading parameters')
#     with tf.Graph().as_default():
#         gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
#         sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
#         with sess.as_default():
#             pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
#     # 加载图片路径
#     tmp_image_paths=copy.copy(image_paths)
#     img_list = []
#     for image in tmp_image_paths:
#         # print(image)

#         # 读取图片 （os.path.expanduser(image)将路径中的～替换为用户）
#         img = misc.imread(os.path.expanduser(image), mode='RGB')

#         # 获取图片的shape
#         img_size = np.asarray(img.shape)[0:2]

#         # 返回边界框数组 （参数分别是输入图片 脸部最小尺寸 三个网络 阈值 factor不清楚）
#         bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

#         # 如果检测出图片中不存在人脸 则将此图路径移除 并输出图中无人脸
#         if len(bounding_boxes) < 1:
#           image_paths.remove(image)
#           print("can't detect face, remove ", image)
#           continue

        
#         det = np.squeeze(bounding_boxes[0,0:4])
#         bb = np.zeros(4, dtype=np.int32)
#         bb[0] = np.maximum(det[0]-margin/2, 0)
#         bb[1] = np.maximum(det[1]-margin/2, 0)
#         bb[2] = np.minimum(det[2]+margin/2, img_size[1])
#         bb[3] = np.minimum(det[3]+margin/2, img_size[0])
#         cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]

#         # 获得剪裁后的脸部，并将其拉伸为固定大小（image_size大小）
#         aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')

#         # 将获得的脸部图片进行白化预处理
#         prewhitened = facenet.prewhiten(aligned)

#         # 将处理后的脸部区域添加入列表
#         img_list.append(prewhitened)
    
#     #将img_list堆叠（增加一维）
#     images = np.stack(img_list)
#     return images








def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    # 创建mtcnn网络，并加载参数
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    # 加载图片路径
    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    
    # print(image)

    # 读取图片 
    img = image_paths

    # 获取图片的shape
    img_size = np.asarray(img.shape)[0:2]

    # 返回边界框数组 （参数分别是输入图片 脸部最小尺寸 三个网络 阈值 factor不清楚）
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    # 如果检测出图片中不存在人脸 则将此图路径移除 并输出图中无人脸
    # if len(bounding_boxes) < 1:
    #     image_paths.remove(image)
    #     print("can't detect face, remove ", image)
    #     continue

    
    det = np.squeeze(bounding_boxes[0,0:4])
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
    bb[3] = np.minimum(det[3]+margin/2, img_size[0])
    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]

    # 获得剪裁后的脸部，并将其拉伸为固定大小（image_size大小）
    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')

    # 将获得的脸部图片进行白化预处理
    prewhitened = facenet.prewhiten(aligned)

    # 将处理后的脸部区域添加入列表
    img_list.append(prewhitened)
    
    #将img_list堆叠（增加一维）
    images = np.stack(img_list)
    return images

with tf.Graph().as_default():
        with tf.Session() as sess:
      
            # 加载模型
            facenet.load_model(model_dir)

            # 训练数据文件夹

            # 返回给定名称的tensor
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")


            # 从训练数据文件夹中加载图片并剪裁，最后embding
            data=load_data('./train_dir/')

            # print(data)
            keys=[]
            for key in data:
                keys.append(key)
                print('folder:{},image numbers：{}'.format(key,len(data[key])))

            print(keys)
            train_x=[]
            train_y=[]

            # 使用mtcnn模型获取每张图中face的数量以及位置，并将得到的embedding数据存储
            for x in data[keys[0]]:
                # print(x)

                images_me = load_and_align_data(x, 160, 44, 1.0)

                feed_dict = { images_placeholder: images_me, phase_train_placeholder:False }
                emb = sess.run(embeddings, feed_dict=feed_dict)
            
                train_x.append(emb)
                train_y.append(0)
            print(len(train_x))


            for y in data[keys[1]]:
                images_others = load_and_align_data(y, 160, 44, 1.0)

                feed_dict = { images_placeholder: images_others, phase_train_placeholder:False }
                emb = sess.run(embeddings, feed_dict=feed_dict)
            
                train_x.append(emb)
                train_y.append(1)
            print(len(train_x))

            print('搞完了，样本数为：{}'.format(len(train_x)))




            

#train/test split
train_x=np.array(train_x)
train_x=train_x.reshape(-1,128)
train_y=np.array(train_y)
print(train_x.shape)
print(train_y.shape)


X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=.3, random_state=42)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

# KNN Classifier  
def knn_classifier(train_x, train_y):  
    from sklearn.neighbors import KNeighborsClassifier  
    model = KNeighborsClassifier()  
    model.fit(train_x, train_y)  
    return model  

classifiers = knn_classifier 

model = classifiers(X_train,y_train)  
predict = model.predict(X_test)  

accuracy = metrics.accuracy_score(y_test, predict)  
print ('accuracy: %.2f%%' % (100 * accuracy)  ) 
  
    
#save model
joblib.dump(model, './models/knn_classifier.model')

model = joblib.load('./models/knn_classifier.model')
predict = model.predict(X_test) 
accuracy = metrics.accuracy_score(y_test, predict)  
print ('accuracy: %.2f%%' % (100 * accuracy)  ) 