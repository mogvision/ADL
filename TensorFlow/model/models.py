import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as KL
from model_profiler import model_profiler

from model.blocks import *
use_bias = False


def Efficient_Unet(in_channels, out_channels, filter_base=32):
    inp = KL.Input(shape=[None, None, in_channels], batch_size=None)
    kwargs = {}#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    yhat_x0, yhat_x2, yhat_x4 = build_denoiser(inp, out_channels, filter_base, kwargs)
    return keras.Model(inputs=inp, 
                        outputs=[yhat_x0, yhat_x2, yhat_x4], 
                        name="Efficient_Unet"
            ) 


def Efficient_Unet_disc(in_channels, filter_base=16):

    inp = KL.Input(shape=[None, None, in_channels], batch_size=None)
    kwargs = {}#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    bridge_map, y_x0_map, y_x2_map, y_x4_map = build_Efficient_Unet_disc(inp, filter_base, kwargs)
    return keras.Model(inputs=inp, 
                        outputs=[bridge_map, y_x0_map, y_x2_map, y_x4_map], 
                        name="Efficient_Unet_disc"
            )





#------------------
def build_denoiser(inp, out_channels, filter_base, kwargs):
    """ Efficient_Unet Architecture """
    f1 = 3*filter_base
    f2 = f1 + filter_base
    f3 = f2 + filter_base
    fb = f3 + filter_base
    #f5 = f4
    #fb = f3

    # Content Enhancer
    s0 = ContentEnhancer(inp, f1, name='ContentEnhancer')

    # Endoder 1, 2, 3
    s1 = residual_block(inp, f1, strides=1, name='encoder1')
    s2 = residual_block(s1,  f2, strides=2, name='encoder2')
    s3 = residual_block(s2,  f3, strides=2, name='encoder3')
    #s4 = residual_block(s3,  f4, strides=2, name='encoder4')
    #s5 = residual_block(s4,  f5, strides=2, name='encoder5')

    # Bridge
    bridge = residual_block(s3, fb, strides=2, name='bridge')

    #x = decoder_block(bridge, s5, f5, name='decoder-1')
    #x = decoder_block(x, s4, f4, name='decoder0')


    # Decoder 1
    x = decoder_block(bridge, s3, f3, name='decoder1')
    # get yhat at scale x4
    yhat_x4 = T3(x, f3, out_channels)


    # Decoder 2
    x = decoder_block(x, s2, f2, name='decoder2')
    # get yhat at scale x2
    yhat_x2 = T2(x, f2, out_channels)


    # Decoder 3
    s1_con = KL.Concatenate(name='decoder3/con_s0_s1')([s1, s0])
    x = decoder_block(x, s1_con, f1,  name='decoder3')
    yhat_x0 = T1(x, f1, out_channels)

    return yhat_x0, yhat_x2, yhat_x4



#------------------
def build_Efficient_Unet_disc(inp, filter_base, kwargs):
    """ Efficient_Unet Architecture """
    name='Efficient_Unet_disc/'

    f1 = filter_base
    f2 = 2*f1
    f3 = 2*f2
    f4 = 2*f3
    f5 = f4
    fb = f5
    
    # Endoder 1, 2, 3
    s1 = residual_block(inp, f1, strides=1, name=name+'/encoder1')
    s2 = residual_block(s1,  f2, strides=2, name=name+'/encoder2')
    s3 = residual_block(s2,  f3, strides=2, name=name+'/encoder3')


    # Bridge
    bridge = residual_block(s3, fb, strides=2, name=name+'/bridge')
    bridge_map = KL.Conv2D(1, 1, padding="same", 
                            activation="sigmoid", use_bias=use_bias, 
                            name='bridge_map')(bridge)

    # Decoder 1
    x = decoder_block(bridge, s3, f3, name=name+'/decoder1')
    # get yhat at scale x4
    yhat_x4_map = T3_disc(x, f3)


    # Decoder 2
    x = decoder_block(x, s2, f2, name=name+'/decoder2')
    # get yhat at scale x2
    yhat_x2_map = T2_disc(x, f2)


    # Decoder 3
    x = decoder_block(x, s1, f1,  name=name+'/decoder3')
    yhat_x0_map = T1_disc(x, f1)


    return bridge_map, yhat_x0_map, yhat_x2_map, yhat_x4_map




if __name__== '__main__':
    in_shape = 3
    out_channels = 3
    kwargs = {}

    model = Efficient_Unet(in_shape, out_channels)
    print(model.outputs[0])
    print(model.outputs[1])
    print(model.outputs[2])



    Batch_size = 1
    profile = model_profiler(model, Batch_size)
    print(profile)


    #print(model.summary())

    #from tensorflow.keras.utils import plot_model 
    #plot_model(model, to_file='model_1.png', show_shapes=True)