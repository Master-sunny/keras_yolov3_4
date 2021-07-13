"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import plot_model
from yolo3.models import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data

# utils.py中的get_random_data被修改成读取单通道图片
def _main():
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #读取数据
    annotation_path = '2007_train.txt'
    log_dir = 'logs/034/'
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'anchor4.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)

    #得到12个锚点
    anchors = get_anchors(anchors_path)
    #输入图像的高和宽 对应4输出做的改动，三输出则为416
    input_shape = (448,448) # multiple of 32, hw
    #判断是否是is_tiny_version
    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2)
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2) # make sure you know what you freeze
    # 参数可视化工具，用于后期画图用的
    logging = TensorBoard(log_dir=log_dir)
    # filename：字符串，保存模型的路径
    # monitor：需要监视的值
    # verbose：信息展示模式，0或1
    # save_best_only：当设置为True时，将只保存在验证集上性能最好的模型
    # mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，
    # 例如，当监测值为val_acc时，模式应为max，
    # 当检测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
    # save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
    # period：CheckPoint之间的间隔的epoch数
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    
    # optimer指的是网络的优化器
    # mode (str) ，可选择‘min’或者‘max’，min表示当监控量停止下降的时候，学习率将减小，max表示当监控量停止上升的时候，学习率将减小。默认值为‘min’
    # factor 学习率每次降低多少，new_lr = old_lr * factor
    # patience=10，容忍网路的性能不提升的次数，高于这个次数就降低学习率
    # verbose（bool） - 如果为True，则为每次更新向stdout输出一条消息。 默认值：False
    # threshold（float） - 测量新最佳值的阈值，仅关注重大变化。 默认值：1e-4
    # cooldown： 减少lr后恢复正常操作之前要等待的时期数。 默认值：0。
    # min_lr,学习率的下限
    # eps ，适用于lr的最小衰减。 如果新旧lr之间的差异小于eps，则忽略更新。 默认值：1e-8。
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)


    # monitor: 监控的数据接口，有’acc’,’val_acc’,’loss’,’val_loss’等等。正常情况下如果有验证集，就用’val_acc’或者’val_loss’。但是因为笔者用的是5折交叉验证，没有单设验证集，所以只能用’acc’了。
    # min_delta：增大或减小的阈值，只有大于这个部分才算作improvement。这个值的大小取决于monitor，也反映了你的容忍程度。例如笔者的monitor是’acc’，同时其变化范围在70%-90%之间，所以对于小于0.01%的变化不关心。加上观察到训练过程中存在抖动的情况（即先下降后上升），所以适当增大容忍程度，最终设为0.003%。
    # patience：能够容忍多少个epoch内都没有improvement。这个设置其实是在抖动和真正的准确率下降之间做tradeoff。如果patience设的大，那么最终得到的准确率要略低于模型可以达到的最高准确率。如果patience设的小，那么模型很可能在前期抖动，还在全图搜索的阶段就停止了，准确率一般很差。patience的大小和learning rate直接相关。在learning rate设定的情况下，前期先训练几次观察抖动的epoch number，比其稍大些设置patience。在learning rate变化的情况下，建议要略小于最大的抖动epoch number。笔者在引入EarlyStopping之前就已经得到可以接受的结果了，EarlyStopping算是锦上添花，所以patience设的比较高，设为抖动epoch number的最大值。
    # verbose: 信息展示模式
    # mode: 就’auto’, ‘min’, ‘,max’三个可能。如果知道是要上升还是下降，建议设置一下。笔者的monitor是’acc’，所以mode=’max’。
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1)
    # 训练和测试的数据按比例分开
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    #seed(10101)中的数值相同每次产生的随机数相同，每一次产生随机数之前都要seed(10101)一次
    #seed()中空每次随机数不同
    np.random.seed(10101)
    #打乱行的顺序
    np.random.shuffle(lines)
    np.random.seed(None)
    #测试数量
    num_val = int(len(lines)*val_split)
    #训练数量
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 1
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=20,
                initial_epoch=0,
                callbacks=[logging, checkpoint])
        model.save(log_dir + 'trained_weights_stage_1.h5')
    print("trained_weights_stage_1 is over")

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        
        # 用于配置训练模型
        # optimizer: 字符串（优化器名）或者优化器实例。 详见 optimizers。
        # loss: 字符串（目标函数名）或目标函数。 详见 losses。 如果模型具有多个输出，则可以通过传递损失函数的字典或列表，在每个输出上使用不同的损失。 模型将最小化的损失值将是所有单个损失的总和。
        # metrics: 在训练和测试期间的模型评估标准。 通常你会使用 metrics = ['accuracy']。 要为多输出模型的不同输出指定不同的评估标准， 还可以传递一个字典，如 metrics = {'output_a'：'accuracy'}。
        # loss_weights: 可选的指定标量系数（Python 浮点数）的列表或字典， 用以衡量损失函数对不同的模型输出的贡献。 模型将最小化的误差值是由 loss_weights 系数加权的加权总和误差。 如果是列表，那么它应该是与模型输出相对应的 1:1 映射。 如果是张量，那么应该把输出的名称（字符串）映到标量系数。
        # sample_weight_mode: 如果你需要执行按时间步采样权重（2D 权重），请将其设置为 temporal。 默认为 None，为采样权重（1D）。 如果模型有多个输出，则可以通过传递 mode 的字典或列表，以在每个输出上使用不同的 sample_weight_mode。
        # weighted_metrics: 在训练和测试期间，由 sample_weight 或 class_weight 评估和加权的度量标准列表。
        # target_tensors: 默认情况下，Keras 将为模型的目标创建一个占位符，在训练过程中将使用目标数据。 相反，如果你想使用自己的目标张量（反过来说，Keras 在训练期间不会载入这些目标张量的外部 Numpy 数据）， 您可以通过 target_tensors 参数指定它们。 它可以是单个张量（单输出模型），张量列表，或一个映射输出名称到目标张量的字典。
        # **kwargs: 当使用 Theano/CNTK 后端时，这些参数被传入 K.function。 当使用 TensorFlow 后端时，这些参数被传递到 tf.Session.run。
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 1 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

        # 使用 Python 生成器（或 Sequence 实例）逐批生成的数据，按批次训练模型。
        # 生成器与模型并行运行，以提高效率。 例如，这可以让你在 CPU 上对图像进行实时数据增强，以在 GPU 上训练模型。
        # keras.utils.Sequence 的使用可以保证数据的顺序， 以及当 use_multiprocessing=True 时 ，保证每个输入在每个 epoch 只使用一次。
        # generator: 一个生成器，或者一个 Sequence (keras.utils.Sequence) 对象的实例， 以在使用多进程时避免数据的重复。 生成器的输出应该为以下之一：
        # 一个 (inputs, targets) 元组
        # 一个 (inputs, targets, sample_weights) 元组。
        # 这个元组（生成器的单个输出）组成了单个的 batch。 因此，这个元组中的所有数组长度必须相同（与这一个 batch 的大小相等）。 不同的 batch 可能大小不同。 例如，一个 epoch 的最后一个 batch 往往比其他 batch 要小， 如果数据集的尺寸不能被 batch size 整除。 生成器将无限地在数据集上循环。当运行到第 steps_per_epoch 时，记一个 epoch 结束。
        # steps_per_epoch: 在声明一个 epoch 完成并开始下一个 epoch 之前从 generator 产生的总步数（批次样本）。 它通常应该等于你的数据集的样本数量除以批量大小。 对于 Sequence，它是可选的：如果未指定，将使用len(generator) 作为步数。
        # epochs: 整数。训练模型的迭代总轮数。一个 epoch 是对所提供的整个数据的一轮迭代，如 steps_per_epoch 所定义。注意，与 initial_epoch 一起使用，epoch 应被理解为「最后一轮」。模型没有经历由 epochs 给出的多次迭代的训练，而仅仅是直到达到索引 epoch 的轮次。
        # verbose: 0, 1 或 2。日志显示模式。 0 = 安静模式, 1 = 进度条, 2 = 每轮一行。
        # callbacks: keras.callbacks.Callback 实例的列表。在训练时调用的一系列回调函数。
        # validation_data: 它可以是以下之一：
        # 验证数据的生成器或 Sequence 实例
        # 一个 (inputs, targets) 元组
        # 一个 (inputs, targets, sample_weights) 元组。
        # 在每个 epoch 结束时评估损失和任何模型指标。该模型不会对此数据进行训练。
        # validation_steps: 仅当 validation_data 是一个生成器时才可用。 在停止前 generator 生成的总步数（样本批数）。 对于 Sequence，它是可选的：如果未指定，将使用 len(generator) 作为步数。
        # class_weight: 可选的将类索引（整数）映射到权重（浮点）值的字典，用于加权损失函数（仅在训练期间）。 这可以用来告诉模型「更多地关注」来自代表性不足的类的样本。
        # max_queue_size: 整数。生成器队列的最大尺寸。 如未指定，max_queue_size 将默认为 10。
        # workers: 整数。使用的最大进程数量，如果使用基于进程的多线程。 如未指定，workers 将默认为 1。如果为 0，将在主线程上执行生成器。
        # use_multiprocessing: 布尔值。如果 True，则使用基于进程的多线程。 如未指定， use_multiprocessing 将默认为 False。 请注意，由于此实现依赖于多进程，所以不应将不可传递的参数传递给生成器，因为它们不能被轻易地传递给子进程。
        # shuffle: 是否在每轮迭代之前打乱 batch 的顺序。 只能与 Sequence (keras.utils.Sequence) 实例同用。
        # initial_epoch: 开始训练的轮次（有助于恢复之前的训练）。
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),#训练数据
            steps_per_epoch=max(1, num_train//batch_size),#
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),#验证数据
            validation_steps=max(1, num_val//batch_size),
            epochs=40,
            initial_epoch=20,#从哪个epoch开始
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])#回调函数
        model.save(log_dir + 'trained_weights_final.h5')
    print("trained_weights_final is over")
    # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    #strip() 移除字符串头尾指定的字符序列。
    class_names = [c.strip() for c in class_names]
    return class_names

#得到12个锚点
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)
#变成下面这种形式 9X2
# [[ 10.  13.]
#  [ 16.  30.]
#  [ 33.  23.]
#  [ 30.  61.]
#  [ 62.  45.]
#  [ 59. 119.]
#  [116.  90.]
#  [156. 198.]
#  [373. 326.]]

#anchors表示12个锚点(x,y)
def create_model(input_shape, anchors, num_classes, load_pretrained=False, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    #这里的锚点数量为12个
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:64, 1:32, 2:16, 3:8}[l], w//{0:64, 1:32, 2:16, 3:8}[l], \
        num_anchors//4, num_classes+5)) for l in range(4)]

    # 经过网络后输出y1、y2、y3、y4 并且是以Model的形式进行返回的
    model_body = yolo_body(image_input, num_anchors//4, num_classes)
    # plot_model(model_body,to_file='model_auth.png',show_shapes=True)
    print("model_body =",model_body.output[3][...])
    print("y_true =",y_true[3])
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        # 使用keras自带的模型加载
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    #这里使用匿名函数，把yolo_loss命名为'yolo_loss'
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=False, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape # (416,416)
    num_anchors = len(anchors)

    # Input():用来实例化一个keras张量
    # keras张量是来自底层后端（Theano或Tensorflow）的张量对象，我们增加了某些属性，使我们通过知道模型的输入和输出来构建keras模型。
    # 添加的keras属性有：1)._keras_shape:整型的形状元组通过keras-side 形状推理传播  2)._keras_history: 最后一层应用于张量，整个图层的图可以从那个层，递归地检索出来。
    # #参数：
    # shape: 形状元组（整型），不包括batch size。for instance, shape=(32,) 表示了预期的输入将是一批32维的向量。
    # batch_shape: 形状元组（整型），包括了batch size。for instance, batch_shape=(10,32)表示了预期的输入将是10个32维向量的批次。
    # name: 对于该层是可选的名字字符串。在一个模型中是独一无二的（同一个名字不能复用2次）。如果name没有被特指将会自动生成。
    # dtype: 预期的输入数据类型
    # sparse: 特定的布尔值，占位符是否为sparse
    # tensor: 可选的存在的向量包装到Input层，如果设置了，该层将不会创建一个占位张量。
    # #返回
    # 一个张量
    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    #将任意表达式封装为 Layer 对象。
    # function: 需要封装的函数。 将输入张量作为第一个参数。
    # output_shape: 预期的函数输出尺寸。 只在使用 Theano 时有意义。 可以是元组或者函数。 如果是元组，它只指定第一个维度； 样本维度假设与输入相同： output_shape = (input_shape[0], ) + output_shape 或者，输入是 None 且样本维度也是 None： output_shape = (None, ) + output_shape 如果是函数，它指定整个尺寸为输入尺寸的一个函数： output_shape = f(input_shape)
    # arguments: 可选的需要传递给函数的关键字参数。
    # 输入尺寸
    # 任意。当使用此层作为模型中的第一层时， 使用参数 input_shape （整数元组，不包括样本数的轴）。
    # 输出尺寸
    # 由 output_shape 参数指定 (或者在使用 TensorFlow 时，自动推理得到)
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main()
