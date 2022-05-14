from typing import Any, Callable, Dict, Tuple, Union, Iterable
import torch
from torchvision import transforms
import numpy as np



class ToTensor_fn(object):
    def __call__(self, sample):
        for k in sample.keys():
            if (sample[k] is not False) and (not isinstance(sample[k], str)):
                sample[k] = torch.from_numpy(sample[k])
        return sample


class Normalization(object):
    def __call__(self, sample):
        for k in sample.keys():
            if (sample[k] is not False) and (not isinstance(sample[k], str)):
                sample[k] /= 255.
        return sample
        

class ImRotate90(object):
    def __init__(self, p):
        self.p = p
    
    def __call__(self, sample):
        assert 1>=self.p>=0, 'p is limited in [0 1]'
        p = self.p*100
        
        if np.random.randint(100, size=1)<= p:
            rand_ =  np.random.randint(1,4, size=1)
            for k in sample.keys():
                if (sample[k] is not False) and (not isinstance(sample[k], str)):
                    sample[k] = np.rot90(sample[k], k=rand_, axes=(0, 1)).copy() 
        return sample




class AddGaussianNoise(object):
    def __init__(self, noise_level:Union[Iterable[float],float], Training:bool):
        for noise_ in noise_level:
            assert noise_ >= 0., 'Enter valid noise level!'

        if Training:
            self.noise_level = np.random.uniform(low=noise_level[0], high=noise_level[1], size=(1,))   
        else:
            self.noise_level = np.max(noise_level)# get the maximum noise value for test

    def __call__(self, sample):
        for key in sample.keys():
            if sample['x'] is not False:
                sample['y'] = sample['x'] + np.random.normal(loc=0.0, 
                                                            scale=self.noise_level/255., 
                                                            size=(sample['x'].shape)
                                                            )
        return sample



class ImFlip_lr(object):
    def __init__(self,p):
        self.p = p
        
    def __call__(self, sample):
        
        assert 1>=self.p>=0, 'p is limited in [0 1]'
        p = self.p*100
        if np.random.randint(100, size=1).item()<= p:
            for k in sample.keys():
                if (sample[k] is not False) and (not isinstance(sample[k], str)):
                    sample[k] = np.fliplr(sample[k]).copy() 
        return sample
          


class ImFlip_ud(object):
    def __init__(self,p):
        self.p = p
        
    def __call__(self, sample):
        assert 1>=self.p>=0, 'p is limited in [0 1]'
        p = self.p*100
        
        if np.random.randint(100, size=1).item()<= p:
            for k in sample.keys():
                if (sample[k] is not False) and (not isinstance(sample[k], str)):
                    sample[k] = np.flipud(sample[k]).copy() 
        return sample


    
class Channel_transpose(object):
    def __init__(self, transpose_tuple):
        assert isinstance(transpose_tuple, tuple) and len(transpose_tuple)==3, "Invalid transposed tuple" 
        self.transpose_tuple = transpose_tuple

    def __call__(self, sample):
        for k in sample.keys():
            if (sample[k] is not False) and (not isinstance(sample[k], str)):
                sample[k] = np.transpose(sample[k], self.transpose_tuple) 
        return sample
  

# Transforms =======================================
def Transform_training(noise_level:Union[Iterable[float], float], Training:bool, transpose_tuple=(2,0,1), p=0.5):
    return transforms.Compose([Normalization(), 
                                ImFlip_lr(p), 
                                ImFlip_ud(p), 
                                AddGaussianNoise(noise_level, Training),
                                Channel_transpose(transpose_tuple), 
                                ToTensor_fn(),
        ]) 


def Test_denoising(noise_level:Union[list,float], Training:bool, transpose_tuple=(2,0,1), p=0.5): 
    return transforms.Compose([Normalization(),
                                Channel_transpose(transpose_tuple), 
                                AddGaussianNoise(noise_level, Training), 
                                ToTensor_fn()]
    )
