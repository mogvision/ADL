from typing import Callable, Tuple, Dict, Union
import os
import json
import string
import shutil
from PIL import Image
import tensorflow as tf
import numpy as np 
import re



# Auxiliary funcs==============
def boolean_string(s):
    if s.lower()=='true':
        return True
    else:
        return False

class struct_cls:
    pass


def get_dir(_dir:Callable[[Union[list,str]],str])->list:
    if type(_dir)== str:
        _dir = list([_dir,])
    
    dirs = []
    for item in _dir:
        dirs.append("".join(list(map(lambda c: c if c not in r'[,*?!:"<>|] \\' else '', item))))
    return dirs



'''
def get_dir(_strings:Union[list,str])->list:
    get_dir_map = dict((ord(char), None) for char in r'[,*?!:"<>|]')
    dirs = []
    for _string in _strings:
        dirs.append("".join(_string.translate(get_dir_map)))
    return dirs
'''

def set_gpu_env(gpu_list:list):
    '''Set gpu/cpu mode.'''
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if gpu_list:
        strategy = tf.distribute.MirroredStrategy(devices=gpu_list)
    else:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    gpus_num = strategy.num_replicas_in_sync
    

    # print the list of devices:
    tf.print(f"[i] Number of GPU devices: {gpus_num}")

    for i, gpu in enumerate(tf.config.experimental.list_physical_devices('GPU')):
        tf.print("\tDevice {}:`{}`\tType: {}".format(i, gpu.name, gpu.device_type))

    for i,cpu in enumerate(tf.config.experimental.list_physical_devices('CPU')):
        tf.print("\tlocal host {}:`{}`\tType: {}".format(i, cpu.name, cpu.device_type))


    """
    os environment:
        #0 = all messages are logged (default behavior)
        #1 = INFO messages are not printed
        #2 = INFO and WARNING messages are not printed
        #3 = INFO, WARNING, and ERROR messages are not printed
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    return strategy, gpus_num


def makedirs_fn(*argv):
    path_ = [arg for arg in argv if arg]
    path_ = os.path.join(*path_)
    if not os.path.exists(path_):
        os.makedirs(path_)
        #tf.gfile.MakeDirs(path_)
    return path_



def get_model_summary(model, model_name, short=False):
    if short:
        for layer in model.layers:
            tf.print(layer.name)
    else:
        model._name = model_name
        model.summary()

    #tf.keras.utils.plot_model(model, to_file='model_1.png', show_shapes=False)


def get_MapDataset_len(MapDataset):
    """ ... """
    # tf.data.experimental.cardinality(item).numpy())
    return MapDataset.cardinality().numpy() 


# Write some images to a folder
def imwrite_Dataset(DS, step, dst, num_channels):
    """ ... """
    if not os.path.exists(dst):
        os.makedirs(dst)

    def normalizeTo255(x):
        """ normalize data between 0 and 255 """ 
        x *= 255.
        return x.astype(np.uint8)


    counter = 0
    if isinstance(DS, dict):
        for item in DS:
            if counter%step == 0:
                if num_channels > 1:
                    Image.fromarray(normalizeTo255(item.numpy())). \
                    convert("RGB").save(os.path.join(dst, str(counter)+'.jpg'))
                else:
                    Image.fromarray(normalizeTo255(item.numpy())). \
                    save(os.path.join(dst, str(counter)+'.jpg'))

            counter += 1
    else:
        for _ in DS.take(get_MapDataset_len(DS)):
            item = next(iter(DS))
            if counter%step == 0:
                im = item[0]

                if num_channels > 1:
                    Image.fromarray(normalizeTo255(im[:,:,0:3].numpy())). \
                    convert("RGB").save(os.path.join(dst, str(counter)+'_awgn.jpg')) 

                    Image.fromarray(normalizeTo255(item[-1].numpy())). \
                    convert("RGB").save(os.path.join(dst, str(counter)+'.jpg')) 
                else:
                    Image.fromarray(normalizeTo255(im[:,:,0].numpy())). \
                    save(os.path.join(dst, str(counter)+'_awgn.jpg')) 

                    Image.fromarray(normalizeTo255(item[-1].numpy())). \
                    save(os.path.join(dst, str(counter)+'.jpg')) 
            counter += 1


# Configuration ==================
def read_config(config_dir):
    STEPS={}
    with open(config_dir, 'rb') as file:
        config = json.load(file)
    #config.get('AnyParams', anyValues???)
    
    STEPS.update({
        'val_per_step':50,
        'test_per_step':50,
        'checkpoint_per_step':50,
        'disc_per_step':3
        })

    config.update({'STEPS':STEPS})
    return config


def environ_setup(parameter, initial_val):
    if parameter in os.environ:
        return os.environ[parameter]
    else:
        return initial_val

def rmtree_dirs(dst_dir):
    if os.path.exists(dst_dir):
        try:
            shutil.rmtree(dst_dir)
            tf.print(f"'{dst_dir}' removed successfully!")
        except OSError as error:
            tf.print(error)
            tf.print(f" '{dst_dir}' can not be removed!")



def make_or_restore_model(model, step_num, ckpt_dir, options, prefix='checkpoint-'):
    """ Either restore the latest model, or load a fresh model if 
    there is no checkpoint available."""

    # check the selected #ckpt, if not exist, find the most recent one
    
    if step_num > 0:
        model_path = f"{ckpt_dir}/{prefix}{step_num}/"
        if not os.path.exists(model_path):
            step_ids = [int(dir_.split(prefix)[-1]) for dir_ in os.listdir(ckpt_dir)]
            step_num = max(step_ids)
            model_path = f"{ckpt_dir}/{prefix}{step_num}/"
    
        loaded = tf.keras.models.load_model(model_path, options=options.load_options)
        tf.print(f"[i] Restoring the model from step {step_num}.")
        return loaded

    tf.print("[i] Creating a new model.")
    return model



def restore_last_model(ckpt_dir, option_load, prefix='checkpoint-'):
    """ restore the latest model"""

    step_ids = [int(dir_.split(prefix)[-1]) for dir_ in os.listdir(ckpt_dir)]
    step_num = max(step_ids)
    model_path = f"{ckpt_dir}/{prefix}{step_num}/"
    tf.print(f"[i] Restoring the model from step {step_num}.")

    return tf.keras.models.load_model(model_path, compile=False, options=option_load)






def store_model(model_name, ckpt_dir, dst_dir, step_num, load_save_device=None):
    """ ... """

    option_load  = tf.saved_model.LoadOptions(experimental_io_device=load_save_device)
    options_save = tf.saved_model.SaveOptions(experimental_io_device=load_save_device)

    model_path = f"{ckpt_dir}/checkpoint-{step_num}/"

    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path, compile=False, options=option_load)
        tf.print(f"[i] Restoring the model from step {step_num}.")

        # save as a new name in the given dir
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
    
        model.save(f'{dst_dir}/{model_name}-{step_num}', options=options_save)




# load only a model
def load_model(model_name, stored_model_dir, step_num, compile_ = False ):
    model_file = os.path.join(stored_model_dir, f'{model_name}-{str(step_num)}.h5')
    return tf.keras.models.load_model(model_file, compile=compile_)


# load checkpoints
def init_models(model, model_name, optimizer, stored_model_dir, step_num ):
    step = tf.Variable(0, dtype=tf.int64)
    checkpoint = tf.train.Checkpoint( model=model, optimizer=optimizer, step=step)
    checkpoint_file = os.path.join(stored_model_dir, f'{model_name}-{str(step_num)}.h5')
    checkpoint.restore(checkpoint_file).expect_partial()


# Extract avg and std of ALL test images
def get_overall_results(csv_filename):

    csv_file = pd.read_csv(csv_filename)  
    sgima_vals = np.unique(csv_file['sigma'])
    PSNR = {}
    SSIM = {}

    for sgima in sgima_vals:
        filt = csv_file.where(csv_file['sigma']==sgima)
        psnr = [filt['psnr'].mean(), filt['psnr'].std()]
        ssim = [filt['ssim'].mean(), filt['ssim'].std()]
        PSNR.update({sgima:psnr})
        SSIM.update({sgima:ssim})
    return PSNR, SSIM



def check_ds(ds, ds_size):

    ds_size = tf.convert_to_tensor(ds_size)
    iter_data = iter(ds)

    counter, step = tf.Variable(1., dtype=tf.float32), tf.Variable(0, dtype=ds_size.dtype)

    while  step < ds_size:
        try:
            x,y,l = next(iter_data)
            step.assign_add(1)
        except StopIteration:
            break
