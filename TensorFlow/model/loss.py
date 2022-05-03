import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops

def BCE (label_smoothing):
    return tf.keras.losses.BinaryCrossentropy(
                    from_logits=False , 
                    label_smoothing=label_smoothing,
                    reduction=tf.keras.losses.Reduction.NONE)


MSE  = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
MSLE = tf.keras.losses.MeanSquaredLogarithmicError(reduction=tf.keras.losses.Reduction.NONE)
MAE_obj = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

# --------------------------------------------
# L1 loss
# --------------------------------------------
#@tf.function()
def MAE(x_gt, yhat, batch_size):
    B = x_gt.get_shape()[0]
    x_ravel = tf.reshape(x_gt, [B,-1])
    y_ravel = tf.reshape(yhat, [B,-1])

    #per_batch_loss = tf.reduce_mean(tf.abs(yhat - x_gt), axis=[1,2,3])
    per_batch_loss = MAE_obj(x_ravel, y_ravel)#/tf.cast(tf.reduce_prod(x_gt), dtype=tf.float32)
    return tf.nn.compute_average_loss(per_batch_loss, global_batch_size=batch_size)


# --------------------------------------------
#                   Hist loss
# --------------------------------------------
#@tf.function()
def Hist_loss(x_gt, yhat, batch_size, hist_loss_fn='log_cosh'):

    #@tf.function()
    def Losses(x_hist, y_hist, loss_name):
        #loss_fn = tf.keras.losses.get(loss_name)()
        #return loss_fn(x_hist, y_hist, reduction=tf.keras.losses.Reduction.NONE)
        l = tf.keras.losses.LogCosh(reduction=tf.keras.losses.Reduction.NONE)
        return l(x_hist, y_hist)



    #@tf.function()
    def tf_histogram(img, B):
        values_range = tf.constant([0., 1.], dtype = tf.float32)
        img = tf.clip_by_value(img, 0., 1.)
        img_vec = tf.reshape(img, [B, -1])
        hist = tf.stack([tf.histogram_fixed_width(img_vec[b,...], 
            values_range, 256 ) for b in range(B)])
        return tf.cast(hist, dtype=img.dtype) #tf.divide(tf.cast(hist, dtype=img.dtype), tf.cast(dim, dtype=img.dtype))  

    """
    Args:
      ?: ?.
    """
    B, n_channels = x_gt.get_shape()[0], x_gt.get_shape()[-1]
    loss_ = tf.zeros((B,), dtype=tf.float32)
    for i in tf.range(n_channels):
        x_hist = tf_histogram(x_gt[...,i], B)
        y_hist = tf_histogram(yhat[...,i], B)

        loss_+= Losses(x_hist, y_hist, hist_loss_fn)  # log_cosh  MeanAbsoluteError  loss_ += tf.keras.losses.log_cosh(x_hist, y_hist)

    dim = tf.math.reduce_prod(x_gt.get_shape())
    
    per_batch_loss = tf.divide(loss_, tf.cast(dim, dtype=tf.float32))  
    return tf.nn.compute_average_loss(per_batch_loss, global_batch_size=batch_size) 



#-------------------------------------------------------------
#                           Edge Loss
#-------------------------------------------------------------
#@tf.function()
def pyr_Loss(x_gt, yhat, batch_size, levels=3):
    """Cha"""

    def atw_kernel(image, ker_base, level=1):
        image_shape = array_ops.shape(image)

        zeros_len = -1 + 2**(level-1)
        ker_len = zeros_len * 4 + len(ker_base)
        kernel_1d = np.zeros((ker_len,), dtype=np.float32)
        kernel_1d[::zeros_len+1] = ker_base/np.sum(ker_base)
        kernel_2d = np.tensordot(kernel_1d, np.transpose(kernel_1d), axes=0)

        # convert To tensor
        kernel_tf = tf.convert_to_tensor(kernel_2d, dtype=image.dtype)
        pad_size  = kernel_tf.get_shape()[0]//2

        kernels_tf = tf.expand_dims(tf.expand_dims(kernel_tf, axis=-1), axis=-1)
        kernels_tf = array_ops.tile( kernels_tf, 
                        [1, 1, image_shape[-1], 1], name='lapalcian_filter')

        return  kernels_tf, pad_size

    def _convolve(image, ker_base, level): 
        # get ATW filter
        kernels_tf, pad_sz=atw_kernel(image, ker_base, level)

        # Use depth-wise convolution to calculate edge maps per channel.
        pad_sizes = [[0, 0], [pad_sz, pad_sz], [pad_sz, pad_sz], [0, 0]]
        padded = array_ops.pad(image, pad_sizes, mode='REFLECT')
    
        #print(image.shape, padded.shape, kernels_tf.shape)
        # Output tensor has shape [batch_size, h, w, d * num_kernels].
        strides = [1, 1, 1, 1]
        output = tf.nn.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')
    
        # Reshape to [batch_size, h, w, d].
        image_shape = array_ops.shape(image)
        shape = array_ops.concat([image_shape, ], 0)
        output = array_ops.reshape(output, shape=shape)
        return output

    B = x_gt.get_shape()[0]
    per_batch_loss = tf.constant(0.,dtype=x_gt.dtype)

    ker_base = [0.002566, 0.1655, 0.6638, 0.1655, 0.002566]
    x_blur = _convolve(x_gt, ker_base, 1)
    y_blur = _convolve(yhat, ker_base, 1)
    
    ker_base = [1., 4., 6., 4., 1.]
    for i in tf.range(1, levels+1):
        x_blur_cur = _convolve(x_blur, ker_base, i)
        y_blur_cur = _convolve(y_blur, ker_base, i)

        # get detail
        Di_x = x_blur - x_blur_cur
        Di_y = y_blur - y_blur_cur

        # update x, y
        x_blur = x_blur_cur
        y_blur = y_blur_cur

        x_ravel = tf.reshape(Di_x, [B,-1])
        y_ravel = tf.reshape(Di_y, [B,-1])
        per_batch_loss = MAE_obj(x_ravel, y_ravel)

    per_batch_loss = tf.divide(per_batch_loss, tf.constant(levels,dtype=tf.float32))
    return tf.nn.compute_average_loss(per_batch_loss, global_batch_size=batch_size)
