import tensorflow as tf 
import numpy as np 
from keras import backend as K 
from yolo import YOLO, detect_video
from PIL import Image
import os
os.system('cd C:/Users/Administrator/gprMax')
image = Image.open('C:/Users/Administrator/keras-yolo3-master/2019.jpg')
# print("image.mode:",image.mode)
# print("channels:",len(image.split()))
image=image.convert("L")
# print("L_image.mode:",L_image.mode)
# print("channels:",len(L_image.split()))
image.show()
# L_image.show()
# path1 = "C:/Users/Administrator/keras-yolo3-master/VOCdevkit/VOC2007/JPEGImages"
# path2 = "C:/Users/Administrator/keras-yolo3-master/VOCdevkit/VOC2007/S-JPEGImages"
# os.makedirs(path2)
# listdirs = os.listdir(path1)
# for p in listdirs:
#     image = Image.open(path1+'/'+p)
#     L_image=image.convert("L")
#     L_image.save(path2+'/'+p)
# a = np.array([[[0,1,2],[0,2,1],[0,3,5]],[[1,1,2],[1,4,7],[1,3,0]]])
# b = np.array([[1,3,4],[2,5,7]])
# anchor_maxes = a / 2.
# anchor_mins = -anchor_maxes
# print(anchor_mins)
# output_path = 'model_data/yolo_conv1.h5'
#splitext用来分离文件和扩展名
# path_01='E:\STH\Foobar2000\install.log'
# path_02='E:\STH\Foobar2000'
# res_01=os.path.splitext(path_01)
# res_02=os.path.splitext(path_02)
# print(res_01)
# print(res_02)
# output_root = os.path.splitext(output_path)[0]
# print(output_path)
# def main():
#     image = Image.open('street.jpg')
#     r_image = YOLO.detect_image(image)
#     r_image.show()
# grid_y = K.tile(K.reshape(K.arange(0, stop=13), [-1, 1, 1, 1]),
#         [1, 13, 1, 1])#1,13,1,1
# grid_x = K.tile(K.reshape(K.arange(0, stop=13), [1, -1, 1, 1]),
#         [13, 1, 1, 1])#13,1,1,1
# a = K.variable(np.array([[1,2,3],[2,3,4]]))
# b = K.variable(np.array([[5,6,7],[6,7,8]]))

# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     grid = K.concatenate([a, b])
#     print(grid)
# if __name__ == '__main__':
#     main()
# a = np.array([[[1,2,3,4,5,6],[2,3,4,5,6,7]],[[5,2,9,4,0,6],[8,5,4,7,6,0]]])
# a = np.array([1,2,3,4])
# print(a[-1])
# y = np.expand_dims(a,axis=0)
# z = np.expand_dims(a,axis=1)
# print(a.shape)
# print(y.shape)
# print(z.shape)
# print(a)
# print(y)
# print(z)
# b = np.arange(16).reshape(4,4)
# model_path = 'model_data/yolo.h5'
# #format用来填充，把format内容填入{}中
# print('{} model, anchors, and classes loaded.'.format(model_path))
# print(b)
# print(b[...,0])#
# print(a[:2])#切片，第一个数省略则表示从0开始
# print(a[...,1])#冒号表示在不知道有多少情况下全部选中，1表示第一列的内容
# print(a[...,:2])
# print(a[...,:3])
# print(a[...,4])
# print(a[...,5])
# print(a[...])
# print(a[::2])
# print("hello world")