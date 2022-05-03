from argparse import ArgumentParser
import os
import time
import datetime
import shutil
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


from util import utils
from model import trainer, loss, metrics
from util.DataLoader import DataLoader
from model.denoiser_trainer import Denoiser_Trainer
from model.discriminator_trainer import Discriminator_Trainer
from model.adl_trainer import ADL_Trainer


dir_path = os.path.dirname(os.path.abspath(__file__))


# tf, cuda, cudnn version
from tensorflow.python.platform import build_info as build
print(f"TensorFlow Version: {tf.__version__}")
print(f"Cuda Version: {build.build_info['cuda_version']}")
print(f"Cudnn version: {build.build_info['cudnn_version']}")

# available devices
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices(),"*"*10)


###################################
parser = ArgumentParser()
parser.add_argument('--DENOISER',  type=str, default='Efficient-Unet', action="store", required=True, help="The network")
parser.add_argument('--train-dirs', nargs="+", default='trainDir', required=True,  help="The datasets can be separated by comma")
parser.add_argument('--test-dirs',  nargs="+", default='TestDir',  required=True, help="The datasets can be separated by comma")
parser.add_argument('--EXPERIMENT', type=str, default='', action="store", required=True, help="The name of experiment")
parser.add_argument('--gpus-list', type=list, default=[], action="store", 
                    help=" list of gpus, e.g. ['/device:GPU:0', '/device:GPU:1']. \
                    Empty list will automatically detect the available gpus.")
parser.add_argument('--json-file', type=str, action="store", default='', required=True, help="configuration file")
parser.add_argument('--channels-num', type=int, action="store", default=3, required=True, help="The number of input/output channels")
parser.add_argument('--DEBUG', choices=('True','False'), default='False', help="Use DEBUG for finding bad images")
args=parser.parse_args()



class struct_cls:
    pass

class Train(object):
    def __init__(self,
                strategy, gpus_num, 
                args, config, 
                loss_denoiser,
                eval_params_denoiser,
                loss_weights_denoiser):

        # Device config
        self.strategy = strategy
        self.gpus_num = gpus_num
        self.args = args

        # Data
        self.config_data = config['data']
        self.args.DEBUG = utils.boolean_string(self.args.DEBUG)

        # Denoiser
        denoiser = struct_cls()
        denoiser.loss = loss_denoiser
        denoiser.loss_weights = loss_weights_denoiser
        denoiser.eval_params  = eval_params_denoiser

        denoiser.config = config['denoiser']
        denoiser.config.update(config['STEPS'])
        denoiser.bs_per_gpu = self.config_data['batch_size_per_gpu']

        denoiser.model = args.DENOISER
        denoiser.config['model'] = args.DENOISER
        denoiser.ckpt_dir = utils.makedirs_fn(dir_path, args.EXPERIMENT, 'denoiser', 'checkpoints')
        denoiser.log_dir  = utils.makedirs_fn(dir_path, args.EXPERIMENT, 'denoiser', 'logs') 
        self.denoiser = denoiser


        # Discriminator
        disc = struct_cls()
        disc.config = config['discriminator']
        disc.config.update(config['STEPS'])
        disc.bs_per_gpu = self.config_data['batch_size_per_gpu'] 
        disc.model = disc.config['model']
        disc.ckpt_dir = utils.makedirs_fn(dir_path, args.EXPERIMENT, 'discriminator', 'checkpoints')
        disc.log_dir  = utils.makedirs_fn(dir_path, args.EXPERIMENT, 'discriminator', 'logs') 
        self.disc = disc


        # ADL
        adl = struct_cls()
        adl.model = config['model']
        adl.config = config['ADL']
        adl.config.update(config['STEPS'])
        adl.bs_per_gpu = self.config_data['batch_size_per_gpu'] 

        adl.results_dir = utils.makedirs_fn(dir_path, args.EXPERIMENT)
        adl.ckpt_dir = utils.makedirs_fn(dir_path, args.EXPERIMENT, 'ADL', 'checkpoints')
        adl.log_dir  = utils.makedirs_fn(dir_path, args.EXPERIMENT, 'ADL', 'logs') 
        self.adl = adl


    def __call__(self):
        load_save_device = self.config_data['localhost'] if self.config_data['localhost'] is not None else None


        # Load Data =============
        dataLoader_obj = DataLoader(strategy=self.strategy,
                                    gpus_num=self.gpus_num,
                                    channels_num=self.args.channels_num,
                                    config=self.config_data, 
                                    train_ds_dir=self.args.train_dirs, 
                                    test_ds_dir=self.args.test_dirs,
                                    debug=self.args.DEBUG)
        ds_train, ds_val, ds_test, steps_per_epoch = dataLoader_obj()

        if self.args.DEBUG:
            utils.check_ds(ds_train, steps_per_epoch['TRAIN'])
            utils.check_ds(ds_val, steps_per_epoch['VAL'])
            utils.check_ds(ds_test, steps_per_epoch['TEST'])


        # Warmup the denoiser network =============
        denoiser_module= Denoiser_Trainer(self.strategy, 
                                        self.gpus_num, 
                                        self.denoiser, 
                                        self.args, 
                                        self.args.channels_num, 
                                        steps_per_epoch, 
                                        load_save_device)
        denoiser_module(ds_train, ds_val, ds_test)
 

        # Warmup the discriminator network =============
        disc_module= Discriminator_Trainer(self.strategy, 
                                            self.gpus_num, 
                                            self.disc, 
                                            self.args, 
                                            self.args.channels_num, 
                                            steps_per_epoch, 
                                            load_save_device)
        disc_module(ds_train)              
    

        # Train ADL =============
        adl_module= ADL_Trainer(self.strategy, 
                                self.gpus_num, 
                                self.adl, 
                                self.denoiser, 
                                self.disc, 
                                self.args, 
                                self.args.channels_num, 
                                steps_per_epoch, 
                                load_save_device)
        adl_module(ds_train, ds_val, ds_test)

        # remove Warmups
        utils.rmtree_dirs(f'{dir_path}/{self.args.EXPERIMENT}/denoiser')
        utils.rmtree_dirs(f'{dir_path}/{self.args.EXPERIMENT}/discriminator')



if __name__== '__main__':
    # Set gpu devices
    strategy, gpus_num = utils.set_gpu_env(args.gpus_list)
    
    # Read configuration file =============
    config = utils.read_config(args.json_file)
    [tf.print(key, item) for key, item in config.items()]
    

    # Evaluation metrics and fidelity term for generator/denoiser
    loss_denoiser_fns = {
        'L1': loss.MAE, 
        'Histgram': loss.Hist_loss, 
        'atw-edge':loss.pyr_Loss
        }

    loss_weights_denoiser = {
        'L1': 1., 
        'Histgram': 1., 
        'atw-edge': 1.
        }

    eval_params_denoiser = {
        'psnr': metrics.PSNR,
        'ssim': metrics.SSIM
    }

    tf.print(f"Let's start training the model", "."*3)
    Trainer = Train(strategy=strategy, 
                    gpus_num=gpus_num,
                    args= args, 
                    config=config, 
                    loss_denoiser=loss_denoiser_fns, 
                    eval_params_denoiser = eval_params_denoiser,
                    loss_weights_denoiser = loss_weights_denoiser)

    ticks = time.time()
    Trainer()
    elapsed_time = time.time() - ticks
    tf.print(f"[i] Training took hh:mm:ss->{str(datetime.timedelta(seconds=elapsed_time))} (hh:mm:ss).") 


    tf.print('Done!')