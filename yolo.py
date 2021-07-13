# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body,yolo_head
from yolo3.utils import letterbox_image
import os
from cv2 import cv2
from keras.utils import multi_gpu_model

# generate中的tiny_yolo_body Input改成了单通道
class YOLO(object):
    #YOLO模型的默认值
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        # self.__dict__.update(self._defaults) # set up default values
        # self.__dict__.update(kwargs) # and update with user overrides
        self.model_path = 'logs/029/trained_weights_final.h5'
        #锚点路径
        self.anchors_path = 'yolo_anchors.txt'
        #种类路径
        self.classes_path = 'model_data/voc_classes.txt'
        #得分阈值
        self.score = 0.5
        #IOU阈值
        self.iou = 0.4
        #图像尺寸
        self.model_image_size = (416, 416)
        #GPU数量
        self.gpu_num = 1
        #输入的内容从generate经过Yolov3的模型获得方框，得分和种类
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)#把path中"~/XXX.txt"包含的"~"和"~user"转换成用户目录
        with open(classes_path) as f:
            #读取文本中每一行的内容
            class_names = f.readlines()
        #从文本中读取的内容含做处理
        class_names = [c.strip() for c in class_names]#strip()用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        print("class_name_list",class_names)
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        #锚点，需要先聚类找到最多出现的几类
        with open(anchors_path) as f:
            #只有一行所以读取
            anchors = f.readline()
        #把这一行的锚点分开并转换成float类型
        anchors = [float(x) for x in anchors.split(',')]
        #转换成9X2类型
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        #加载权重模型
        model_path = os.path.expanduser(self.model_path)
        #首先判断文件格式是否正确
        #断言，‘,’号之前的不符合就输出后面的内容
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        #如果是tiny版本锚点数是6，判断是不是等于6
        is_tiny_version = num_anchors==6 # default setting
        try:
            #加载h5格式的模型
            self.yolo_model = load_model(model_path, compile=False)
        except:
            #如果是tiny版本就调用tiny_yolo_body(),否则调用yolo_body()
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,1)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            #调用完模型调用权重
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:#锚点个数除以（模型输出乘以种类个数加5）

            # #输出结果是[<tf.Tensor 'conv2d_59/BiasAdd:0' shape=(?, ?, ?, 255) dtype=float32>, 
            # #<tf.Tensor 'conv2d_67/BiasAdd:0' shape=(?, ?, ?, 255) dtype=float32>, 
            # #<tf.Tensor 'conv2d_75/BiasAdd:0' shape=(?, ?, ?, 255) dtype=float32>]
            # print((num_classes + 5)) --->85
            # 模型最后一层的形状最后一个数，输出形状的最后一个数
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'
        
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        #生成hsv的元组
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        #生成的元组80个表示有80个类别80个颜色
        #[(0.0, 1.0, 1.0), (0.0125, 1.0, 1.0), (0.025, 1.0, 1.0), (0.0375, 1.0, 1.0), (0.05, 1.0, 1.0), (0.0625, 1.0, 1.0), (0.075, 1.0, 1.0), (0.0875, 1.0, 1.0), (0.1, 1.0, 1.0), (0.1125, 1.0, 1.0), (0.125, 1.0, 1.0), (0.1375, 1.0, 1.0), (0.15, 1.0, 1.0), (0.1625, 1.0, 1.0), (0.175, 1.0, 1.0), (0.1875, 1.0, 1.0), (0.2, 1.0, 1.0), (0.2125, 1.0, 1.0), (0.225, 1.0, 1.0), (0.2375, 1.0, 1.0), (0.25, 1.0, 1.0), (0.2625, 1.0, 1.0), (0.275, 1.0, 1.0), (0.2875, 1.0, 1.0), (0.3, 1.0, 1.0), (0.3125, 1.0, 1.0), (0.325, 1.0, 1.0), (0.3375, 1.0, 1.0), (0.35, 1.0, 1.0), (0.3625, 1.0, 1.0), (0.375, 1.0, 1.0), (0.3875, 1.0, 1.0), (0.4, 1.0, 1.0), (0.4125, 1.0, 1.0), (0.425, 1.0, 1.0), (0.4375, 1.0, 1.0), (0.45, 1.0, 1.0), (0.4625, 1.0, 1.0), (0.475, 1.0, 1.0), (0.4875, 1.0, 1.0), (0.5, 1.0, 1.0), (0.5125, 1.0, 1.0), (0.525, 1.0, 1.0), (0.5375, 1.0, 1.0), (0.55, 1.0, 1.0), (0.5625, 1.0, 1.0), (0.575, 1.0, 1.0), (0.5875, 1.0, 1.0), (0.6, 1.0, 1.0), (0.6125, 1.0, 1.0), (0.625, 1.0, 1.0), (0.6375, 1.0, 1.0), (0.65, 1.0, 1.0), (0.6625, 1.0, 1.0), (0.675, 1.0, 1.0), (0.6875, 1.0, 1.0), (0.7, 1.0, 1.0), (0.7125, 1.0, 1.0), (0.725, 1.0, 1.0), (0.7375, 1.0, 1.0), (0.75, 1.0, 1.0), (0.7625, 1.0, 1.0), (0.775, 1.0, 1.0), (0.7875, 1.0, 1.0), (0.8, 1.0, 1.0), (0.8125, 1.0, 1.0), (0.825, 1.0, 1.0), (0.8375, 1.0, 1.0), (0.85, 1.0, 1.0), (0.8625, 1.0, 1.0), (0.875, 1.0, 1.0), (0.8875, 1.0, 1.0), (0.9, 1.0, 1.0), (0.9125, 1.0, 1.0), (0.925, 1.0, 1.0), (0.9375, 1.0, 1.0), (0.95, 1.0, 1.0), (0.9625, 1.0, 1.0), (0.975, 1.0, 1.0), (0.9875, 1.0, 1.0)]
        #map是会根据提供的函数对指定序列做映射
        #lambda是匿名函数作为map的前一个函数，后面的hsv_tuples映射到x中做运算 colorsys是anaconda自带的颜色管理的程序
        #这里的*x表示元组中的内容
        self.colors = list(map(lambda x: colorsys.hsv_to_gray(*x), hsv_tuples))
        self.colors = list(
            #对归一化的颜色放到0到255的尺度上 list是列表跟数组很像
            map(lambda x: (int(x * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.
        # Generate output tensor targets for filtered bounding boxes.
        #先做图像形状的占位
        self.input_image_shape = K.placeholder(shape=(2, ))
        #多GPU模式
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        #输入参数：模型预测输出
        #print(self.yolo_model.output)
#重点
        #使用yolo_eval来Evaluate YOLO model on given input and return filtered boxes."""
        # box_xy, box_wh, box_confidence, box_class_probs = yolo_head(self.yolo_model.output[0],self.anchors, len(self.class_names), (416,416))
        print('self.yolo_model.output[0]=',K.shape( self.yolo_model.output[0])[...])
        
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)

        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            #先判断图像的尺寸能不能被32整除，也就是图像尺寸是否符合，不符合就报错
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            #首先按比例缩小并补全图片到(416,416)大小，tuple将列表变成元组，reversed把列表的内容倒过来
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            #将图片的尺寸转换成能被32整除的大小，先width再height
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        #将图片转换成numpy数据处理
        boxed_image = boxed_image.convert('L')
        boxed_image = np.expand_dims(boxed_image,2)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        # 像素数值归一化
        image_data /= 255.
        #扩展0维位数 image的大小(1，x，y，3)图片编号，长宽和通道数
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        #
        out_boxes, out_scores, out_classes = self.sess.run(
            #把从generate处获得的数据赋值到三个变量out_boxes, out_scores, out_classes
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                #图片的尺寸要调换一下
                self.input_image_shape: [image.size[1], image.size[0]],
                #学习阶段标志是一个布尔张量（0 = test，1 = train），它作为输入传递给任何 Keras 函数，以在训练和测试时执行不同的行为操作。
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        #np.floor表示向下取整
        #设置字体这是字体的文件font/FiraMono-Medium.otf  字体大小要根据图片大小来定size=np.floor(3e-2 * image.size[1] + 0.5)

        # #方框的粗细要根据图片的尺寸来定
        #enumerate返回的是out_classes的索引和内容
        gray = cv2.cvtColor(np.asarray(image) ,cv2.COLOR_RGB2BGR)
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)

            #返回给定字符串的大小，以像素为单位。最终结果是整个字符串的长度和高度（，47）
            label_size = cv2.getTextSize(label,cv2.FONT_HERSHEY_COMPLEX, 0.4 ,2)

            top, left, bottom, right = box
            #对矩形框的坐标取整top, left和0比取大值 bottom, right和图片的尺寸比取小值
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom),label_size[0][1])

            if top - label_size[0][1] >= 0:
                #使用top - label_size[1]是为了表示文字在方框的上方
                text_origin = [left, top - label_size[0][1]]
            else:
                text_origin = [left, top + 1]

            #用来画文字的底色
            cv2.rectangle(gray,(left, top),(right, bottom),(0,0,255),thickness=2,lineType=2)
            # 位置 内容 字体
            cv2.putText(gray, label, (text_origin[0],text_origin[1]), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255,0,0), 1)
        end = timer()
        print(end - start)
        return gray

    def close_session(self):
        self.sess.close()

#
def detect_video(yolo, video_path, output_path=""):
    
    vid = cv2.VideoCapture(video_path)
    #如果
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    #获得视频的参数
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #判断是否把视频保存
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        #按照读取的视频参数保存视频
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        # cap.read()按帧读取视频，return_value,frame是获cap.read()方法的两个返回值。其中return_value是布尔值，
        # 如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵。
        return_value, frame = vid.read()
        #   If you have an image in NumPy::

        #   from PIL import Image
        #   import numpy as np
        #   im = Image.open('hopper.jpg')
        #   a = np.asarray(im)

        # Then this can be used to convert it to a Pillow image::im = Image.fromarray(a)
        #读取的图片是一个三维数组，使用fromarray把数组转换成图片
        image = Image.fromarray(frame)
        #经过上面的函数处理得到结果
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        #计算处理一帧图片需要的时间
        exec_time = curr_time - prev_time
        #迭代一下，为下次计算时间做准备
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        #时间记满1秒计算出处理的帧数作为fps，然后清除数据为下一次做准备
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        #把fps放到图片上
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        #如果要保存则直接写入到要保存的地址中
        if isOutput:
            out.write(result)
        #如果这个cv2.waitKey(1)每1毫秒检测一次按键输入，如果没有按键输入cv2.waitKey(1)返回-1则不会执行break
        #反之有按键按下且按键是'q'则if条件成立。执行break循环结束。
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()


def main():
    yolo = YOLO()
    image = Image.open('000004.jpg')
    # print(image.mode)
    # image = image.convert("L")
    # image = np.expand_dims(image,2)
    # print(image.shape)
    # print(len(r_image.split()))
    r_image = yolo.detect_image(image)
    cv2.imshow("detect results",r_image)
    cv2.imwrite("logs/029/4.jpg",r_image)
    cv2.waitKey(0)

    # r_image.show()


if __name__ == '__main__':
    main()

