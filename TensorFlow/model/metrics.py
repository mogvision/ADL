import tensorflow as tf
import util.DataLoader as DataLoader

def PSNR(imgs_x, imgs_y, global_batch_size):
    per_batch_loss = tf.image.psnr(imgs_x, imgs_y, max_val=1)
    return tf.nn.compute_average_loss(per_batch_loss, global_batch_size=global_batch_size)


def SSIM(imgs_x, imgs_y, global_batch_size):
    per_batch_loss = tf.image.ssim(imgs_x, imgs_y, max_val=1)
    return tf.nn.compute_average_loss(per_batch_loss, global_batch_size=global_batch_size)



# metrics in np domain
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
class MetricEval(object):
    @staticmethod
    def mse(gt, pred):
        """ Compute Mean Squared Error (MSE) """
        return np.mean((gt - pred) ** 2)

    @staticmethod
    def nmse(gt, pred):
        """ Compute Normalized Mean Squared Error (NMSE) """
        return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2

    @staticmethod
    def psnr(gt, pred):
        """ Compute Peak Signal to Noise Ratio metric (PSNR) """
        #return np.abs(peak_signal_noise_ratio(gt, pred, data_range=gt.max()))
        mse = np.mean((255.*gt - 255.*pred) ** 2)
        if mse == 0:
            return float('inf')
        return 20. * np.log10(255. / np.sqrt(mse))

    @staticmethod
    def ssim(gt, pred):
        """ Compute Structural Similarity Index Metric (SSIM). """
        if len(gt.shape) == 2:
            return np.abs(structural_similarity(gt, pred, channel_axis=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0))
        else:
            avg = 0.0
            for k in range(0, gt.shape[2]):
                avg += np.abs(structural_similarity(gt[:,:,k], pred[:,:,k], channel_axis=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0))

            return avg/gt.shape[2]