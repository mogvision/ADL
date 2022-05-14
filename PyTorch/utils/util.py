from typing import Callable, Tuple, Dict, Union
import os
import json
import shutil
import numpy as np
from math import exp

import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.autograd import Variable
import torch.nn.functional as F
import subprocess



from model import MODELS


# Auxiliary funcs==============
def boolean_string(s):
    if s.lower()=='true':
        return True
    else:
        return False

class struct_cls:
    pass

def makedirs_fn(*argv):
    path_ = [arg for arg in argv if arg]
    path_ = os.path.join(*path_)
    if not os.path.exists(path_):
        os.makedirs(path_)
        #tf.gfile.MakeDirs(path_)
    return path_


# Configuration =========================================
def read_config(config_dir):
    STEPS={}
    with open(config_dir, 'rb') as file:
        config = json.load(file)
    #config.get('AnyParams', anyValues???)
    
    STEPS.update({
        'val_per_step':100,
        'test_per_step':200,
        'checkpoint_per_step':200,
        })

    config.update({'STEPS':STEPS})
    return config


def rmtree_dirs(dst_dir):
    if os.path.exists(dst_dir):
        try:
            shutil.rmtree(dst_dir)
            print(f"'{dst_dir}' removed successfully!")
        except OSError as error:
            print(error)
            print(f" '{dst_dir}' can not be removed!")


# DDP =========================================
def get_dist_info(rank):
    rank, world_size, num_gpus= 0, 1, 1
    num_gpus = torch.cuda.device_count()
    if num_gpus is None:
        warnings.warn('[warning!] GPU not detected!')

    torch.cuda.set_device(rank % num_gpus)
    world_size = int(os.environ["WORLD_SIZE"])
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend="nccl", 
        init_method='tcp://127.0.0.1:2345',
        world_size=world_size,
         rank=rank)
    
    return rank, world_size, num_gpus


def cleanup():
    dist.destroy_process_group()

# prepare dirs =========================================
def prep(param, mode, dir_path, experiment):
    # prepare the directories
    param.ckpt_dir = makedirs_fn(dir_path, experiment, mode, 'checkpoints')
    param.log_dir  = makedirs_fn(dir_path, experiment, mode, 'logs') 

    # prepare writers
    param.writer_numerical_dir = f'{param.log_dir}/numerical'
    param.writer_imgs_dir = f'{param.log_dir}/images'

    return param



# get model =========================================
def save_model(model, file_):
    if isinstance(model,  (DataParallel, DistributedDataParallel)):
        model = model.module

    model_dict = model.state_dict()
    for key, param in model_dict.items():
        model_dict[key] = param.cpu()
    torch.save(model_dict, file_)
    #torch.save(self.model.state_dict(), file_)


def load_model(model, model_path, param_key='params'):
    if isinstance(model,  (DataParallel, DistributedDataParallel)):
        model = model.module

    model_dict = torch.load(model_path)
    if param_key in model_dict.keys():
        model_dict = model_dict[param_key]
    model.load_state_dict(model_dict, strict=True)
    return model


def get_model(model_name, in_channels, out_channels, ckpt_dir, device,
                prefix='ckpts_step_', suffix='.pt'):
        """We get the model and then load its last saved one if avail"""

        def _get_denoiser(model_name, in_channels, out_channels):
            """ get denoiser model"""
            module_ = getattr(MODELS, model_name)
            return module_(in_channels, out_channels)

        # get model
        model = _get_denoiser(model_name, in_channels, out_channels)


        # load the last saved model if avail
        model_path = None
        if os.path.exists(ckpt_dir):
            step_ids = [int(dir_.split(prefix)[-1].split(suffix)[0]) for dir_ in os.listdir(ckpt_dir)]
            if len(step_ids) > 0:
                step_num = max(step_ids)
                model_path = f"{ckpt_dir}/{prefix}{step_num}{suffix}"

        model = load_model(model, model_path, param_key='params')
        print(f"[i] Restoring the model from step {step_num}.")
        return model.to(device)



#----------------------------------------
import socket
import warnings


def find_free_port():
    # Copied from https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/launch.py # noqa: E501
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


#---------------------------
# copied from https://github.com/Po-Hsun-Su/pytorch-ssim

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)