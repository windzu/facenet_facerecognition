# import cv2


# video="http://admin:admin@192.168.0.107:8081/"   #此处@后的ipv4 地址需要修改为自己的地址
#         # 参数为0表示打开内置摄像头，参数是视频文件路径则打开视频
# capture =cv2.VideoCapture(video)

# cv2.namedWindow("camera",1)

# c=0
# num = 0
# frame_interval=3 # frame intervals  

# while True:
#     ret, frame = capture.read()
#     timeF = frame_interval

#     # print(shape(frame))

#     if(c%timeF == 0):
#         find_results=[]
#         # cv2.imshow("camera",frame)
   
#         image1=cv2.putText(frame,'detected:{}'.format(find_results), (50,100), 
#                 cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0 ,0), 
#                 thickness = 2, lineType = 2)
#         cv2.imshow('camera',image1)

#     c+=1

#     key = cv2.waitKey(3)

#     if key == 27:
#         #esc键退出
#         print("esc break...")
#         break

#     if key == ord(' '):
#         # 保存一张图像
#         num = num+1
#         filename = "frames_%s.jpg" % num
#         cv2.imwrite(filename,img)

if 2>0 and 2>0:

    print(111)