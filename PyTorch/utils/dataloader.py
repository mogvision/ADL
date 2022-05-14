from typing import Any, Callable, Dict, Tuple, Union, Iterable
import os
import numpy as np
import random
import re

import torch
import torch.nn as nn
from torch.utils.data  import Dataset, DataLoader
from skimage import io, color, transform
import warnings
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler



from utils.transform_collections import  Transform_training


class DataLoader_cls(Dataset): 
    def __init__(self,
                num_workers: int,
                batch_size:int,
                channels_num: int,
                train_ds_dir: Union[str,list],
                test_ds_dir: Union[str,list],
                config:  Union[str,list],
                distributed:bool
        )->Union[DataLoader, DataLoader, DataLoader]: 
        r"""Data loader using DDP
        
        Args:
            num_workers: number of workers. It will be divided by the number of gpus
            batch_size: the size of batches. It will be divided by the number of gpus
            channels_num: input channels (RGB:3, grey:1)
            train_ds_dir: A list of directories for training datasets
            test_ds_dir: A list of directories for test datasets
            config: configuration file for data
        """

        self.distributed = distributed
        
        # chech the directories
        self.train_dir = _get_dir(train_ds_dir) 
        self.test_dir = _get_dir(test_ds_dir)


        # dataloader params
        self.config = config
        self.shuffle = config['shuffle']
        self.train_valid_ratio = config['train_valid_ratio']
        self.num_valid_max = config['num_valid_max']
        self.random_seed = config['random_seed']


        self.DL_params = {'batch_size': batch_size,
                        'num_workers': num_workers,
                        'pin_memory': config['pin_memory'],
                        'drop_last':  config['drop_last']
                    }

        # x: ground-truth, y: noisy sample (if False, we will add synthesized noise to x)
        self.data_mode = {'y':False, 'x':True, 'mask':False, 'filename':False}
        self.DS_params = {'data_mode': self.data_mode, 
                        'task_mode': config['task_mode'] ,
                        'WHC': [config['W'],config['H'],channels_num], 
                        'img_format': config['img_types'],
                    }

    def __call__(self):
        
        # get train & validation datasets
        img_files = _get_files(self.train_dir, self.config['img_types'], self.data_mode)

        # divide the datasets into train and validation
        train_idx, valid_idx = self._train_valid_sampler(len(img_files['x']))

        if self.distributed:  
            # create Dataset_cls: TRAIN
            img_files_train = {key:[img_files[key][idx] for idx in train_idx] 
                            for key, val in self.data_mode.items() if val}
            Dataset_Train = Dataset_cls(img_files=img_files_train, 
                                    Training=True, 
                                    noise_level=self.config['train_std_interval'], 
                                    **self.DS_params)
            # create Dataset_cls: VALID
            img_files_valid = {key:[img_files[key][idx] for idx in valid_idx] 
                                for key, val in self.data_mode.items() if val}
            Dataset_Valid = Dataset_cls(img_files=img_files_valid, 
                                    Training=True, 
                                    noise_level=self.config['train_std_interval'], 
                                    **self.DS_params)
            # create sampler
            train_sampler =DistributedSampler(Dataset_Train, drop_last=self.config['drop_last'], 
                                seed=self.random_seed, shuffle=self.shuffle)
            valid_sampler = DistributedSampler(Dataset_Valid, drop_last=self.config['drop_last'], 
                                seed=self.random_seed, shuffle=self.shuffle)

            # create data loader
            train_loader = DataLoader(dataset= Dataset_Train, sampler= train_sampler, 
                                    **self.DL_params )
            valid_loader = DataLoader(dataset= Dataset_Valid, sampler= valid_sampler, 
                                    collate_fn=collate_fn, **self.DL_params )

        else:
            # create Dataset_cls
            Dataset_Train = Dataset_cls(img_files=img_files, 
                                    Training=True, 
                                    noise_level=self.config['train_std_interval'], 
                                    **self.DS_params)
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            train_loader = DataLoader(dataset= Dataset_Train, sampler= train_sampler, 
                                    **self.DL_params )
            valid_loader = DataLoader(dataset= Dataset_Train, sampler= valid_sampler, 
                                    collate_fn=collate_fn, **self.DL_params )


        # create dataloader for teste
        img_files = _get_files(self.test_dir, self.config['img_types'], self.data_mode)
        Dataset_Test  = Dataset_cls(img_files=img_files,  
                                    Training=False, 
                                    noise_level=self.config['test_stdVec'], 
                                    **self.DS_params)
        test_loader  = DataLoader(dataset= Dataset_Test, shuffle= False, 
                                    collate_fn=collate_fn, **self.DL_params)

        return  train_loader, valid_loader, test_loader



    def _train_valid_sampler(self, len_train_valid):
        """sampler for training and validation"""

        split = len_train_valid - int(self.train_valid_ratio * len_train_valid)

        # check the upper limit of validation samples
        if split > self.num_valid_max:
            split = self.num_valid_max

        indices = list(range(len_train_valid))

        if self.shuffle:
            torch.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        return train_idx, valid_idx
    

def collate_fn(batch):
    """remove bad samples"""
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

 
def _get_dir(_dir:Callable[[Union[list,str]],str])->list:
    """check and get directories"""
    if type(_dir)== str:
        _dir = list([_dir,])
    
    dirs = []
    for item in _dir:
        dirs.append("".join(list(map(lambda c: c if c not in r'[,*?!:"<>|] \\' else '', item))))
    return dirs


def _initilize_data_mode(data_mode:Dict[str,bool])->Dict[str,Any]:
    """ Ininitlize data mode by the input data mode"""

    data_mode_ = {key: False for key in ['y', 'x', 'mask']} 
    if data_mode is not None:
        data_mode_.update(data_mode)
    
    return data_mode_

def _get_files(dirs_:Union[list,Iterable[str]], img_format, data_mode):
    """ get image files
        in-args:
            dirs_: list of data directories
        out-args: 
            img_dirs: list of the address of all avail images
    """
    dirs_ = list(dirs_)
    img_dirs = _initilize_data_mode(data_mode)

    if data_mode['y']:
        dirs_ = [os.path.join(dir_,'HR') for dir_ in dirs_] 

    img_dirs['x'] = [os.path.join(path, name) 
                    for dir_i in dirs_ 
                        for path, subdirs, files_ in os.walk(dir_i)  
                            for name in files_ 
                                if name.lower().endswith(tuple(img_format))]
    if data_mode['y']:
        img_dirs['y'] =  [files.replace('/HR/', '/LR/') for files in x_files]

    if data_mode['mask']:
        img_dirs['mask'] =  [files.replace('/HR/', '/mask/') for files in x_files]

    return img_dirs



def _im_read_resize(PATH:str, WHC:Tuple[int, Iterable[int]]):
    """read and resize images"""

    assert isinstance(WHC, (list, tuple)) and len(WHC)==3, "Invalid tuple for width, height, channel"
    Width, Height, Channel = WHC[0], WHC[1], WHC[2]

    img = io.imread(PATH).astype(np.float64)
    WH = img.shape[0:2]

    # because of pyramid structure, the data size must be dividable by `blocks`
    blocks = 8
    WH = list(map(lambda x: (x//blocks)*blocks, WH))
    img = transform.resize(img, WH)

    if Channel ==1:
        img = color.rgb2gray(img)   

    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1) 

    if Width > 0 and Height > 0:
        img = transform.resize(img, (Height, Width)) 

    return np.array(img, dtype=np.float64) 




class Dataset_cls(Dataset):
    def __init__(self,
                img_files: str,
                Training: bool,
                data_mode:Dict[str, bool],
                task_mode:str,
                noise_level: Union[list,float],
                WHC:list,
                img_format:list,
                keep_last_n_dirs:int=3
                ):
        r""" Dataset class for one dataset

        Args:
            img_files: lsit of filenames
            Training: whether testing or training
            data_mode: type of input images
            task_mode: DEN=Denoising, etc
            noise_level: the level of noise
            WHC: [Width, height, depth]
            img_format: extension of images
            keep_last_n_dirs: save last n directories of a filename
        """
        super(Dataset, self).__init__()

        self.data_mode = _initilize_data_mode(data_mode)
        self.task_mode = task_mode
        self.noise_level = noise_level
        self.WHC = WHC
        self.Training = Training
        self.img_files = img_files 
        self.keep_last_n_dirs = keep_last_n_dirs
        
    def __len__(self):
        return len(self.img_files['x'])


    def __getitem__(self, index):
        sample_index = _initilize_data_mode(self.data_mode)

        try:
            if self.data_mode['x']:
                sample_index['x'] = _im_read_resize(self.img_files['x'][index], self.WHC)
                sample_index['filename'] = str('/'.join(
                                self.img_files['x'][index].rsplit('/', self.keep_last_n_dirs)[1:]
                                ))   

            # check the size of images.
            if (sample_index['x'].ndim < 3) or (sample_index['x'].shape[2] != self.WHC[2]):
                return None

            if self.data_mode['y']:
                sample_index['y'] = _im_read_resize(self.img_files['y'][index], self.WHC)
        
            if self.data_mode['mask']:
                sample_index['mask'] = _im_read_resize(self.img_files['mask'][index], self.WHC)  

            if self.task_mode == 'DEN':
                T = Transform_training(self.noise_level, self.Training)
                return T(sample_index)
            else:
                raise NotImplementedError 

        except IOError:
            pass