"""MobileNet v3"""


from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, BatchNormalization, Add, Multiply, Reshape
from tensorflow.keras import backend as K



def relu6(x):
    """Relu 6
    """
    return K.relu(x, max_value=6.0)

def hard_swish(x):
    """Hard swish
    """
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0

def return_activation(x, nl):
    """Convolution Block
    This function defines a activation choice.

    # Arguments
        x: Tensor, input tensor of conv layer.
        nl: String, nonlinearity activation type.

    # Returns
        Output tensor.
    """
    if nl == 'HS':
        x = Activation(hard_swish)(x)
    if nl == 'RE':
        x = Activation(relu6)(x)

    return x

def conv_block(inputs, filters, kernel, strides, nl):
    """Convolution Block
    This function defines a 2D convolution operation with BN and activation.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        nl: String, nonlinearity activation type.

    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)

    return return_activation(x, nl)

def SE(inputs):
    """Squeeze and Excitation.
    This function defines a squeeze structure.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
    """
    input_channels = int(inputs.shape[-1])

    x = GlobalAveragePooling2D()(inputs)
    x = Dense(input_channels, activation='relu')(x)
    x = Dense(input_channels, activation='hard_sigmoid')(x)
    x = Reshape((1, 1, input_channels))(x)
    x = Multiply()([inputs, x])

    return x

def bottleneck(inputs, filters, kernel, e, s, squeeze, nl,alpha=1.0):
    """Bottleneck
    This function defines a basic bottleneck structure.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        e: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        squeeze: Boolean, Whether to use the squeeze.
        nl: String, nonlinearity activation type.
        alpha: Integer, width multiplier.

    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    input_shape = K.int_shape(inputs)

    tchannel = int(e)
    cchannel = int(alpha * filters)

    r = s == 1 and input_shape[3] == filters

    x = conv_block(inputs, tchannel, (1, 1), (1, 1), nl)

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = return_activation(x, nl)

    if squeeze:
        x = SE(x)

    x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = Add()([x, inputs])

    return x
