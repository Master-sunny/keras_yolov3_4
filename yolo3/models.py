"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import numpy as np
import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from yolo3.utils import compose

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
#wraps本身就是一个装饰器，因为它返回的是一个“函数”即partial对象，这个对象接收函数作为参数，同时以函数作为返回值。
#@wraps(Conv2D)
# filters：卷积核的数目（即输出的维度）
# kernel_size：单个整数或由两个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
# strides：单个整数或由两个整数构成的list/tuple，为卷积的步长。如为单个整数，则表示在各个空间维度的相同步长。任何不为1的strides均与任何不为1的dilation_rate均不兼容
# padding：补0策略，为“valid”, “same” 。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
# activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
# dilation_rate：单个整数或由两个个整数构成的list/tuple，指定dilated convolution中的膨胀比例。任何不为1的dilation_rate均与任何不为1的strides均不兼容。
# data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。
# use_bias:布尔值，是否使用偏置项
# kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
# bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
# kernel_regularizer：施加在权重上的正则项，为Regularizer对象
# bias_regularizer：施加在偏置向量上的正则项，为Regularizer对象
# activity_regularizer：施加在输出上的正则项，为Regularizer对象
# kernel_constraints：施加在权重上的约束项，为Constraints对象
# bias_constraints：施加在偏置上的约束项，为Constraints对象
def DarknetConv2D(*args, **kwargs):  #kwargs是字典里面包含了后面要用到的strides
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    #任意组合多个函数，从左到右求值。
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def resblock_body(x, num_filters, num_blocks):
    #应该是上采样构成残差网络
    #num_blocks表示要操作的次数
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    #PCBL
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    #这里的strides是(2,2)所以经过这一层之后特征图的长宽变成原来的二分之一
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    #CBLR
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        #y的输出的层数和大小与x的都是一样的所以张量里面的量可以直接相加
        x = Add()([x,y])
    return x
def res_conv(x, num_filters):
    # x = ZeroPadding2D(((1,0),(1,0)))(x)
    #这里的strides是(2,2)所以经过这一层之后特征图的长宽变成原来的二分之一
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        #y的输出的层数和大小与x的都是一样的所以张量里面的量可以直接相加
    x = Add()([x,y])
    return x
#draknet网络主体
def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    #416X416X3经过这一层（卷积核是3X3）之后变成416X416X32
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    #经过这一层之后变成208X208X64
    x = resblock_body(x, 64, 1)
    #经过这一层之后变成104X104X128
    x = resblock_body(x, 128, 2)
    #经过这一层之后变成52X52X256 输出52X52
    x = resblock_body(x, 256, 8)
    #经过这一层之后变成26X26X512 输出26X26
    x = resblock_body(x, 512, 8)
    #经过这一层之后变成13X13X1024 输出13X13
    x = resblock_body(x, 1024, 4)
    return x
#中间输出层，x要上采样拼接，y要直接输出
def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1,1)))(x)
    return x, y

#最终的yolo输出：预选框的个数*（5+种类数）
# def yolo_body(inputs, num_anchors, num_classes):
#     """Create YOLO_V3 model CNN body in Keras."""
#     darknet = Model(inputs, darknet_body(inputs))
#     #y1输出
#     x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))
#     #x经过卷积上采样与152层的输出拼接得到y2
#     x = compose(
#             DarknetConv2D_BN_Leaky(256, (1,1)),
#             UpSampling2D(2))(x)
#     x = Concatenate()([x,darknet.layers[152].output])
#     x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))
#     #x卷积上采样与92层拼接
#     x = compose(
#             DarknetConv2D_BN_Leaky(128, (1,1)),
#             UpSampling2D(2))(x)
#     x = Concatenate()([x,darknet.layers[92].output])
#     x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))
#     #最后输出y1y2y3
#     return Model(inputs, [y1,y2,y3])

# 尝试简化版的yolov3，不使用多次的重复卷积 428X428
def yolo_body(inputs, num_anchors, num_classes):
    x = DarknetConv2D_BN_Leaky(16, (3,3))(inputs)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
    #经过这一层之后变成208X208X64
    x = res_conv(x,32)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
    #经过这一层之后变成104X104X128
    x = res_conv(x, 64)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
    #经过这一层之后变成52X52X256 输出52X52
    x1 = res_conv(x, 128)
    x2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x1)
    #经过这一层之后变成26X26X512 输出26X26
    x2 = res_conv(x2, 256)
    x3 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x2)
    #13X13
    x3 = res_conv(x3, 512)
    x4 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x3)

    x4 = res_conv(x4, 512)
    # x3 = MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(x3)

    x4 = res_conv(x4,512)
    x4 = DarknetConv2D_BN_Leaky(256, (1,1))(x4)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x4)
    x4 = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x4)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x3,x4])
    x3 = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x3)
    y3 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x3])
    x2 = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x2)
    y4 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x1,x2])
    return Model(inputs, [y1,y2,y3,y4])

# def yolo_body(inputs, num_anchors, num_classes):
#     x = DarknetConv2D_BN_Leaky(16, (3,3))(inputs)
#     x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
#     #经过这一层之后变成208X208X64
#     x = res_conv(x,32)
#     x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
#     #经过这一层之后变成104X104X128
#     x = res_conv(x, 64)
#     x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
#     #经过这一层之后变成52X52X256 输出52X52
#     x1 = res_conv(x, 128)
#     x1 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x1)
#     #经过这一层之后变成26X26X512 输出26X26
#     x2 = res_conv(x1, 256)
#     x2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x2)
#     #13X13
#     x3 = res_conv(x2, 512)
#     x3 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x3)

#     x3 = res_conv(x3, 512)
#     x3 = MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(x3)

#     x3 = res_conv(x3,512)
#     x3 = DarknetConv2D_BN_Leaky(256, (1,1))(x3)
#     y1 = compose(
#             DarknetConv2D_BN_Leaky(512, (3,3)),
#             DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x3)
#     x2 = compose(
#             DarknetConv2D_BN_Leaky(256, (1,1)),
#             UpSampling2D(2))(x3)
#     y2 = compose(
#             Concatenate(),
#             DarknetConv2D_BN_Leaky(256, (3,3)),
#             DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x3,x2])
#     x1 = compose(
#             DarknetConv2D_BN_Leaky(256, (1,1)),
#             UpSampling2D(2))(x1)
#     y3 = compose(
#             Concatenate(),
#             DarknetConv2D_BN_Leaky(256, (3,3)),
#             DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])
#     return Model(inputs, [y1,y2,y3])

#改二：将所有的卷积深度减小为原来的1/2
# 卷积深度从16 32 64 128 256 -->8 16 32 64 128
def tiny_yolo_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3)))(inputs)
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            DarknetConv2D_BN_Leaky(256, (1,1)))(x1)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])
# def tiny_yolo_body(inputs, num_anchors, num_classes):
#     '''Create Tiny YOLO_v3 model CNN body in keras.'''
#     x = DarknetConv2D_BN_Leaky(16, (3,3))(inputs)
#     x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
#     #经过这一层之后变成208X208X64
#     x = res_conv(x,32)
#     x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
#     #经过这一层之后变成104X104X128
#     x = res_conv(x, 64)
#     x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
#     #经过这一层之后变成52X52X256 输出52X52
#     x = res_conv(x, 128)
#     x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
#     #经过这一层之后变成26X26X512 输出26X26
#     x1 = res_conv(x, 256)
#     x2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x1)
#     x2 = res_conv(x2, 256)
#     x2 = MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(x2)
#     x2 = res_conv(x2,512)
#     x2 = DarknetConv2D_BN_Leaky(256, (1,1))(x2)
#     y1 = compose(
#             DarknetConv2D_BN_Leaky(512, (3,3)),
#             DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

#     x2 = compose(
#             DarknetConv2D_BN_Leaky(128, (1,1)),
#             UpSampling2D(2))(x2)
#     y2 = compose(
#             Concatenate(),
#             DarknetConv2D_BN_Leaky(256, (3,3)),
#             DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

#     return Model(inputs, [y1,y2])


#这是对yolo_body的输出进行解码，求比例映射到原来的图片中
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    #使用最后层的特征来限定边界的参数
    """Convert final layer features to bounding box parameters."""
    #1,13,13,3,85
    num_anchors = len(anchors)#每一个特征层有三个预选框
    # Reshape to batch, height, width, num_anchors, box_params.
    #把anchors变成一个张量，内部形状分别对应batch_size,格点数（13），格点数（13），先验框的数量，宽高[1, 1, 1, 3, 2]
    #也就是13X13的每一个格点都有三个先验框中心坐标，然后是宽高信息，没有目标的宽高为0
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])
    #得到格点的形状13X13
    grid_shape = K.shape(feats)[1:3] # height, width
    #构建一个13X13的格点                0到13展开13,13,1,1                       height表示图像高度
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])#1,13,1,1
    #tile函数是把K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]沿着第一维复制grid_shape[0]次，展开成13,13,1,1
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])#13,1,1,1
    #拼接成13,13,1,2，以列表的形式对应网格中的点[[0,0],[0,1],...,[1,0]]
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))#改变数据类型
    #batch_size,13,13,3,85
    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
    #feats中的内容最终是num_classes+5个，25或者85，分别对应x、y、w、h、confidence和classes
    #所以下面的的操作都是对feats的内容操作
    # Adjust preditions to each spatial grid point and anchor size.
    #因为grid就是13,13,1,2
    #feats[..., :2]中前两个是xy的偏移量除以13 这里吧grid_shape中的元素倒序然后把数据类型转换成feats的类型
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    #宽和高也需要调整 乘以预选框的数值再输入图像的长宽，也是求的比例
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5]) #置信度
    box_class_probs = K.sigmoid(feats[..., 5:]) #对每一个种类的预测
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    #调换位置
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    #在带灰边的图片中的目标方框转换成原来图片比例的方框
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    #输出边框和边框的得分值
    #得到的是每个特征层的的方框
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    #把特征层的方框转换成图片中的方框
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs #是目标的先被布尔操作变成1，不是目标的是0.然后再乘以置信度就表示这个类的得分值
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    #特征层数量，普通为3个 13X13 26X26 52X52
    num_layers = len(yolo_outputs)
    #特征层1对应678 是13X13大小的
    #特征层2对应345 是26X26大小的
    #特征层1对应012 是52X52大小的
    anchor_mask = [[9,10,11],[6,7,8], [3,4,5], [0,1,2]] if num_layers==4 else [[3,4,5], [1,2,3]] # default setting
    #416X416  13X32=416
    input_shape = K.shape(yolo_outputs[0])[1:3] * 64
    boxes = []
    box_scores = []
    for l in range(num_layers):#对每一个特征层都做处理
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
        #anchor_mask一次性取出三个供每个特征层使用
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    #对于每一类都找出最合适的
    for c in range(num_classes): 
        # TODO: use keras backend instead of tf.
        #这是对预测大于阈值取1小于阈值取0的操作
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        #这是对预测大于阈值取1小于阈值取0的操作
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        #非极大值抑制
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        #K.gather中在给定的张量class_box_scores中搜索给定下标的向量。
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    #拼接在一起就是一个图片中的所有目标
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)
    
    return boxes_, scores_, classes_

#true_boxes是2007_train.txt文件下的真实方框，input_shape是(416,416)
#anchors变成下面这种形式 9X2
# [[ 10.  13.]
#  [ 16.  30.]
#  [ 33.  23.]
#  [ 30.  61.]
#  [ 62.  45.]
#  [ 59. 119.]
#  [116.  90.]
#  [156. 198.]
#  [373. 326.]]
#num_classes是'model_data/voc_classes.txt'里的类型
#此函数用来训练时  供data_generator调用
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting特征层数量一般3
    anchor_mask = [[9,10,11],[6,7,8], [3,4,5], [0,1,2]] if num_layers==4 else [[3,4,5], [1,2,3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    #得到中心点坐标
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    #得到矩形的长宽
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    #除以416得到比例
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]
    #true_boxes的维度是框的数量乘以5（x1,y1,x2,y2,class）
    m = true_boxes.shape[0]
    # 13,26,52
    grid_shapes = [input_shape//{0:64, 1:32, 2:16, 3:8}[l] for l in range(num_layers)]
    #mX13X13X3X85
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true


def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    '''Return yolo_loss tensor
    #输入：
    #args：是一个list 组合，包括了预测值和真实值，具体如下：
    #     args[:num_layers]--预测值yolo_outputs，
    #     args[num_layers:]--真实值y_true，
    #     yolo_outputs，y_true同样是list，分别是[y1,y2,y3]三个feature map 上的的预测结果,
    #     每个y都是m*grid*grid*num_anchors*(num_classes+5),作者原文是三层，分别是(13,13,3,25)\
    #     (26,26,3,25),(52,52,3,25)
    #anchors:输入预先选择好的anchors box，原文是9个box,三层feature map 各三个。
    #num_classes：原文分了20类  
    #ignore_thresh=.5:如果一个default box与true box的IOU 小于ignore_thresh， 
    #                 则作为负样本confidence 损失。
    #print_loss：loss的打印开关。
    #输出：一维张量。
    # 

    '''
    #len(anchors)=12 层数是4
    #总共应该有8层，4层网络输出，4层是预测输出
    num_layers = len(anchors)//3 # default setting
    #前三是特征层的输出
    yolo_outputs = args[:num_layers]
    #后三个是y_true这样才能计算损失
    y_true = args[num_layers:]
    # print("args = ",args)
    # print("num_layers = ",num_layers)
    #              52X52     26X26    13X13
    anchor_mask = [[9,10,11],[6,7,8], [3,4,5], [0,1,2]] if num_layers==4 else [[3,4,5], [1,2,3]]
    #448X448
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 64, K.dtype(y_true[0]))
    # 7X7 14X14 28X28 56X56 总共四层
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    #这里的m是指有多少个
    m = K.shape(yolo_outputs[0])[0] # batch_size, 一次处理几张图片
    mf = K.cast(m, K.dtype(yolo_outputs[0]))
    #l表示的是每一层的输出。总共有四层 7X7 14X14 28X28 56X56
    for l in range(num_layers):
        #表示置信度
        object_mask = y_true[l][..., 4:5]
        #所有的种类
        true_class_probs = y_true[l][..., 5:]
        #这里的yolo_head用来计算损失与之前获得真实值不同
        # print("anchors[anchor_mask[l]] = ",anchors[anchor_mask[l]])
        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        #将中心坐标和宽高拼接一起变成预测的box
        # print("raw_pred = ",raw_pred)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        #把y_true的值也转换成中心坐标和宽高的形式和上面的pred_box格式相同
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        
        raw_true_wh = K.log(y_true[l][..., 2:4]/ anchors[anchor_mask[l]] * input_shape[::-1])
        #这部操作是避免出现log(0) = 负无穷，故当object_mask置信率接近0是返回全0结果
        #K.switch根据object_mask的值来决定输出raw_true_wh还是K.zeros_like(raw_true_wh)说白了四目标就赋值，不是目标就清零
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # Find ignore mask, iterate over each of batch.
        #size=1表示二维动态的张量，可以往里面写入读取
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')#将真实标定的数据置信率转换为T or F的掩膜
        #用来处理一层的所有预测最佳窗口
        def loop_body(b, ignore_mask):
            #object_mask_bool(b,13,13,3,4)--五维数组，第b张图的第l层feature map.
            #true_box将第b图第l层feature map,有目标窗口的坐标位置取出来。true_box[x,y,w,h]
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])#y_true中哪一层从b个维度开始取前四个参数
            
            iou = box_iou(pred_box[b], true_box)#单张图片单个尺度算iou，即该层所有预测窗口
            #pred_box(13,13,3,4)与真实窗口true_box(设有j个)之间的IOU，输出为iou(13,13,3,j)
            best_iou = K.max(iou, axis=-1)#先取每个grid上多个anchor box上的最大的iou
            #向ignore_mask指定b索引写入元素
            #删掉小于阈值的BBOX,ignore_mask[b]存放的是pred_box(13,13,3,4)iou小于
            #ignore_thresh的grid，即ignore_mask[b]=[13,13,3],如果小于ignore_thresh，其值为0;
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])#对所有图片循环，得到ignore_mask[b][13,13,3]
        #将tf.TensorArray中元素叠成一个张量
        ignore_mask = ignore_mask.stack()#将一个列表中维度数目为R的张量堆积起来形成维度为R+1的新张量,R应该就是b。
        ignore_mask = K.expand_dims(ignore_mask, -1)#ignore_mask的shape是(b,13,13,3,1)
        #当一张box的最大IOU低于ignore_thresh，则作为负样本参与计算confidence 损失。
        #这里保存的应该是iou满足条件的BBOX

        # K.binary_crossentropy is helpful to avoid exp overflow.
        #xy损失 object_mask表示是目标的才处理，不是就不处理
        # print("raw_true_xy=",raw_true_xy)
        # print("raw_pred[...,0:2]=",raw_pred[...,0:2])
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        #wh损失
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
        #置信度损失
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        #类别损失
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)
        #求均值
        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss
