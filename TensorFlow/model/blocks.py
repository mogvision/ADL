import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as KL

ks = 3
use_bias = False
leaky_relu_alpha=0.1

def batchnorm_relu(inputs, name='bn_relue'):
    """ Batch Normalization & ReLU """
    x = KL.BatchNormalization(name=name+'/bn')(inputs)
    x = KL.Activation("relu", name=name+'/relu')(x)
    return x


def ContentEnhancer(inputs, num_filters, strides=1, name='ContentEnhancer'):
    """ Content Enhancer """

    conv1 = KL.Conv2D( num_filters//2, 7, dilation_rate = [1,1], use_bias=use_bias, padding='same', name=name+'/conv1')
    conv2 = KL.Conv2D( num_filters//2, 7, dilation_rate = [2,2], use_bias=use_bias, padding='same', name=name+'/conv2')
    conv4 = KL.Conv2D( num_filters//2, 7, dilation_rate = [4,4], use_bias=use_bias, padding='same', name=name+'/conv3')
    x = KL.Concatenate(name=name+'/con1')([conv1(inputs), conv2(inputs), conv4(inputs)])
    x = batchnorm_relu(x, name=name+'/bn_relue')
    x = KL.Conv2D(num_filters, 7, padding="same", strides=1, use_bias=use_bias, name=name+'/conv4')(x)
    x = KL.BatchNormalization(name=name+'/bn2')(x)

    # Shortcut Connection
    s = KL.Conv2D(num_filters, 1, padding="same", strides=1,use_bias=use_bias, name=name+'/conv5')(inputs)
    
    # Addition
    x += s

    # Activation
    #x = KL.Activation("relu", name=name+'/relu2')(x)
    x = tf.nn.leaky_relu(x, alpha=leaky_relu_alpha,  name=name+'/relu2')

    return x



def decoder_block(inputs, skip_features, num_filters, name='decoder_block'):
    """ Decoder Block """
    initializer = tf.random_normal_initializer(0., 0.02)


    #x = KL.UpSampling2D((2, 2), name=name+'/up')(inputs)
    x = KL.Conv2DTranspose(num_filters, ks, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False,
                                    name=name+'/up')(inputs)

    x = KL.Concatenate(name=name+'/con')([x, skip_features])
    x = residual_block(x, num_filters, strides=1, name=name+'/residual_block')
    return x




def residual_block(inputs, num_filters, strides=1, name='residual_block'):
    """ Convolutional Layers """
    x = KL.Conv2D(num_filters, ks, padding="same", strides=strides, use_bias=use_bias, name=name+'/conv1')(inputs)
    x = batchnorm_relu(x, name=name+'/bn1_relue')
    x = KL.Conv2D(num_filters, ks, padding="same", strides=1, use_bias=use_bias, name=name+'/conv2')(x)
    x = KL.BatchNormalization(name=name+'/bn2')(x)

    # Shortcut Connection
    s = KL.Conv2D(num_filters, 1, padding="same", strides=strides, use_bias=use_bias, name=name+'/conv3')(inputs)

    # Addition
    x += s

    # Activation
    x = KL.Activation("relu", name=name+'/relu2')(x)
    return x




# --------------- build transformers

def T1(inputs, num_filters, out_channels, strides=1, name='T1'):
    x = residual_block(inputs, num_filters//2, strides=1, name=name+'/out_x0')
    yhat = KL.Conv2D(out_channels, 1, padding="same", activation="sigmoid", use_bias=use_bias, name='conv4')(x)
    return yhat


def T2(inputs, num_filters, out_channels, strides=1, name='T2'):
    x = residual_block(inputs, num_filters//2, strides=1, name=name+'/out1_x0')
    x = residual_block(x, num_filters//4, strides=1, name=name+'/out1_x2')
    yhat = KL.Conv2D(out_channels, 1, padding="same", activation="sigmoid", use_bias=use_bias, name=name+'/conv4')(x)
    return yhat


def T3(inputs, num_filters, out_channels, strides=1, name='T3'):
    x = residual_block(inputs, num_filters//2, strides=1, name=name+'/out1_x0')
    x = residual_block(x, num_filters//4, strides=1, name=name+'/out1_x2')
    x = residual_block(x, num_filters//8, strides=1, name=name+'/out1_x4')
    yhat = KL.Conv2D(out_channels, 1, padding="same", activation="sigmoid", use_bias=use_bias, name=name+'/conv4')(x)
    return yhat


def T1_disc(inputs, num_filters, strides=1, name='T1_disc'):
    x = residual_block(inputs, num_filters//2, strides=1, name=name+'/out_x0')
    x = tf.nn.leaky_relu(x, alpha=leaky_relu_alpha,  name=name+'/leakyrelue')
    yhat = KL.Conv2D(1, 1, padding="same", use_bias=use_bias, name=name+'/conv4')(x)
    return yhat


def T2_disc(inputs, num_filters, strides=1, name='T2_disc'):
    x = residual_block(inputs, num_filters, strides=1, name=name+'/out1_x0')
    x = residual_block(x, num_filters//2, strides=1, name=name+'/out1_x2')
    x = tf.nn.leaky_relu(x, alpha=leaky_relu_alpha,  name=name+'/leakyrelue')
    yhat = KL.Conv2D(1, 1, padding="same", use_bias=use_bias, name=name+'/conv4')(x)
    return yhat


def T3_disc(inputs, num_filters, strides=1, name='T3_disc'):
    x = residual_block(inputs, num_filters,    strides=1, name=name+'/out1_x0')
    x = residual_block(x, num_filters//2, strides=1, name=name+'/out1_x2')
    x = residual_block(x, num_filters//4, strides=1, name=name+'/out1_x4')
    x = tf.nn.leaky_relu(x, alpha=leaky_relu_alpha,  name=name+'/leakyrelue')
    yhat_map = KL.Conv2D(1, 1, padding="same", use_bias=use_bias, name=name+'/conv4')(x)
    return yhat_map