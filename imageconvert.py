import cv2
from scipy import misc
import os
from os.path import join as pjoin

# date={}
# for guy in os.listdir('./train_dir/pic_me'):   
    
#     curr_pics = cv2.imread(pjoin('./train_dir/pic_me',guy))
#     crop=misc.imresize(curr_pics, (160, 160), interp='bilinear') 
#     cv2.imwrite('160'+guy,crop)


dir='/home/wind/下载/lfw'
num=0
for i in os.listdir(dir):
    temp_dir=pjoin(dir,i)
    for f in os.listdir(temp_dir):
        temp_pic=cv2.imread(pjoin(temp_dir,f)) 
        cv2.imwrite(f,temp_pic)
    num=num+1
    if num>=100:
        break