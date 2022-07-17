import os
from   argparse import ArgumentParser
from   tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import pathlib
import time

from util.utils import read_config
from util.DataLoader_test import DataLoader
from model import metrics
from util import utils

root_dir = os.path.dirname(os.path.abspath(__file__))

parser = ArgumentParser()
parser.add_argument('--saved_model', type=str, default='testCV', action="store", required=True, help="the saved model`s directory")
parser.add_argument('--test-dirs', type=str, default='testCV', action="store", required=True, help="test directory")
parser.add_argument('--num-channels', type=int, default=-1, action="store", required=True, help="number of channels")
parser.add_argument('--gpu', type=str, dest='gpu', default='0')
parser.add_argument('--EXPERIMENT', type=str, default='', action="store", required=True, help="name of experiment")
parser.add_argument('--json-file', type=str, action="store", default='', required=True, help="configuration file")
parser.add_argument('--save-images', choices=('True','False'), default='True', help="save the results as image")

args=parser.parse_args()



def TFstring2str(TFstring):
    return str(TFstring.numpy()).split("'")[1]


class Test(object):
    def __init__(self, 
                config, 
                store_dir,
                csv_filename):

        # Model
        self.model = config['model']
        self.csv_filename = csv_filename
        config['data']['num_channels'] = args.num_channels
        # Data
        self.DataLoader_obj = DataLoader( config = config['data'], 
                                        test_ds_dir = args.test_dirs)

        # save_dir 
        self.store_dir = store_dir
        if not os.path.exists(self.store_dir):
            os.makedirs(self.store_dir)

    def __call__(self):
        test_DSs = self.DataLoader_obj() 

        # Get model
        load_save_device = '/job:localhost'
        option_load  = tf.saved_model.LoadOptions(experimental_io_device=load_save_device)
        if args.saved_model is None:
            ckpt_dir = os.path.join(root_dir, args.EXPERIMENT,  'ADL/checkpoints') 
        else:
            ckpt_dir = os.path.join(root_dir, args.saved_model) 

        model = utils.restore_last_model(ckpt_dir, option_load)

        # Create directories for storing results
        for ds_name, DSs in test_DSs.items():
            for distortion_name, DS in DSs.items():
                result_dir = os.path.join(self.store_dir, ds_name, distortion_name)
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)


        # type of to-be-stored images
        im_type= ".png"

        PSNR, SSIM, IMG_NAME, SIGMA, Time =[], [], [], [], []
        for ds_name, DSs in test_DSs.items():
            print(f"dataset: {ds_name}..." )
            
            for distortion_name, DS in DSs.items():
                print(f"\tdistortion type: {distortion_name}...")
                sigma = int(float(distortion_name.split('_wgn_')[-1]))
                result_dir = os.path.join(self.store_dir, ds_name, distortion_name)

                psnr, ssim, img_name_tmp, sigma_tmp = [], [], [], []
                # Loop over the dataset
                im_ind = 0
                for x, y, img_name in DS.batch(1):

                    start_time=time.time()
                    y1, y2, y3 = model.predict(x)       
                    timee =  time.time() - start_time  
                    Time.append(timee)    

                    y_hat = tf.identity(y1)

                    # save output images
                    head, tail = os.path.split(TFstring2str(img_name))
                    im_name =  f'{pathlib.Path(tail).stem}'  
                    img_name_tmp.append(im_name)
                    sigma_tmp.append(sigma)



                    if utils.boolean_string(args.save_images): 
                        file_name = os.path.join(result_dir, im_name + "_input" + im_type)
                        tf.keras.preprocessing.image.save_img(file_name, x[0,...] )

                        file_name = os.path.join(result_dir, im_name + "_gt" + im_type)
                        tf.keras.preprocessing.image.save_img(file_name, y[0,...] )


                        file_name = os.path.join(result_dir, im_name + "_ADL_pred" + im_type)
                        tf.keras.preprocessing.image.save_img(file_name, y1[0,...] )

                        #file_name = os.path.join(result_dir, im_name + "_ADL_pred_x2" + im_type)
                        #tf.keras.preprocessing.image.save_img(file_name, y2[0,...] )

                        #file_name = os.path.join(result_dir, im_name + "_ADL_pred_x4" + im_type)
                        #tf.keras.preprocessing.image.save_img(file_name, y3[0,...] )


                    # apply evaluation metrics
                    y_np = np.squeeze(y.numpy()).astype(np.float32)
                    y_hat_np = np.squeeze(y_hat.numpy()).astype(np.float32)

                    eval_psnr = metrics.MetricEval.psnr(y_np, y_hat_np)
                    psnr.append( eval_psnr.astype(np.float32) )

                    eval_ssim = metrics.MetricEval.ssim(y_np, y_hat_np)
                    ssim.append( eval_ssim.astype(np.float32) )

                    tf.print( f'sigma:{sigma}- img_name: {img_name}, \
                        psnr: {eval_psnr.astype(np.float32):0.3f}dB, \
                        ssim: {eval_ssim.astype(np.float32):0.3f}, \
                        time: {timee:0.3f}sec')

                    im_ind += 1


                PSNR.append(psnr)
                SSIM.append(ssim)
                IMG_NAME.append(img_name_tmp)
                SIGMA.append(sigma_tmp)


        IMG_NAME = np.array(IMG_NAME).ravel()
        SIGMA = np.array(SIGMA).ravel()
        PSNR = np.array(PSNR).ravel()
        SSIM = np.array(SSIM).ravel()

        pd.DataFrame({'name': IMG_NAME, 
                'sigma': SIGMA, 
                'psnr': PSNR, 
                'ssim': SSIM }
                ).to_csv(self.csv_filename, index=True)

        # save seprately as npy file
        for sigma in np.unique(SIGMA):
            ind = SIGMA == sigma
            psnr, ssim = PSNR[ind], SSIM[ind]
            np.savez(  os.path.join(self.store_dir, 
                        f"{args.EXPERIMENT}_ADL_sigma_{sigma}"), 
                        psnr=psnr, ssim=ssim)

        # Elap. time
        Time = np.array(Time).ravel()
        AvgTime=np.mean(Time)
        print("Avg. Elap. Time: --- %s seconds ---" % (AvgTime))

if __name__== '__main__':

    # Set gpu/cpu mode.
    ######### Set GPUs ###########
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if int(args.gpu) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


    # Configuration =============
    config = utils.read_config(args.json_file)
    [print(key, item) for key, item in config.items()]
    


    store_dir = os.path.join(root_dir, f'{args.EXPERIMENT}_results')
    csv_filename = os.path.join(store_dir, f'ADL{args.EXPERIMENT}.csv')

    tf.print("Let's start evaluating the results...")
    Tester=Test( config=config, 
                store_dir=store_dir,
                csv_filename=csv_filename)
    Tester()

    print('Done!')