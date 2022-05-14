from typing import Any, Callable, Dict, Tuple, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data  import DataLoader

from model import trainer
from model import MODELS
from utils.util import load_model

class Denoiser_Trainer(object):
    def __init__(self,
                denoiser:Dict,
                device:Union[str, int],
                in_channels:int, 
                out_channels:int,
                writer_numerical:SummaryWriter, 
                writer_imgs:SummaryWriter,
                distributed,
        )->None:
        r"""Denoiser trainer
        
        Args:
            denoiser: current rank
            device: cuda device no.
            in_channels: input channels (RGB:3, grey:1)
            out_channels: output channels (RGB:3, grey:1)
        """
        super(Denoiser_Trainer, self).__init__() 

        self.device = device
        self.rank =device if distributed else 0
        self.distributed = distributed

        self.epochs = denoiser.config['epochs']
        self.lr = denoiser.config['lr']

        self.metrics = denoiser.eval_params
        self.print_model = denoiser.config['print_model']

        self.loss = denoiser.loss
        self.loss_weights = denoiser.loss_weights

        self.write_path = denoiser.log_dir
        self.checkpoints_path = denoiser.ckpt_dir

        self.validation_per_step = denoiser.config['val_per_step']
        self.test_per_step = denoiser.config['test_per_step']
        self.ckpts_per_step = denoiser.config['checkpoint_per_step']

        # get and distribute the model 
        self.model = self._get_model('denoiser', denoiser.model, 
                            in_channels, out_channels, 
                            self.checkpoints_path)

        # get optimizer
        self.optimizer, self.scheduler = self._get_optimizer(self.model, 
                                            denoiser.config['optimizer'], 
                                            denoiser.config['lr_scheduler'], 
                                            denoiser.config['lr'])

        # get writers
        self.writer_numerical = writer_numerical
        self.writer_imgs = writer_imgs

    
    def __call__(self, 
                ds_train_loader:DataLoader, 
                ds_valid_loader:DataLoader, 
                ds_test_loader:DataLoader 
                )->bool:
        if self.rank==0:
            print("[i] Configuring denoiser...")
        trainer = Denoiser_Trainer_module(rank =self.rank, 
                                        model = self.model,
                                        device = self.device,
                                        lr = self.lr,
                                        writer_numerical = self.writer_numerical,
                                        writer_imgs = self.writer_imgs,
                                        checkpoints_path = self.checkpoints_path,
                                        print_model = self.print_model)
        if self.rank==0:
            print("[i] Compiling denoiser...")
        trainer.compile( optimizer = self.optimizer, 
                        scheduler = self.scheduler, 
                        loss = self.loss, 
                        loss_weights = self.loss_weights, 
                        metrics = self.metrics,
                        distributed=self.distributed)

        if self.rank==0:
            print("[i] Fitting denoiser...")
        trainer.fit(train_data = ds_train_loader, 
                    validation_data = ds_valid_loader, 
                    test_data = ds_test_loader, 
                    epochs = self.epochs, 
                    validation_per_step = self.validation_per_step,
                    test_per_step = self.test_per_step, 
                    ckpts_per_step = self.ckpts_per_step)
        return True



    def _get_denoiser(self, model_name, in_channels, out_channels):
        """ get denoiser model"""
        module_ = getattr(MODELS, model_name)
        return module_(in_channels, out_channels)


    def _get_model(self, mode, model_name, in_channels, out_channels, ckpt_dir, 
                prefix='ckpts_step_', suffix='.pt'):
        """We get the model and then load its last saved one if avail"""

        # get model
        model = self._get_denoiser(model_name, in_channels, out_channels)

        # load the last saved model if avail
        model_path = None
        if os.path.exists(ckpt_dir):
            step_ids = [int(dir_.split(prefix)[-1].split(suffix)[0]) for dir_ in os.listdir(ckpt_dir)]
            if len(step_ids) > 0:
                step_num = max(step_ids)
                model_path = f"{ckpt_dir}/{prefix}{step_num}{suffix}"

        if model_path:
            model = load_model(model, model_path)
            print(f"[i] {mode}-GPU{self.rank}: Restoring the model from step {step_num}.")
        else:
            print(f"[i] {mode}-GPU{self.rank}: Creating a new model.")

        if self.distributed:
            # map model into DDP
            model = model.cuda(self.device)
            model = DistributedDataParallel(model, device_ids=[self.device]) 
        else:
            model = model.to(self.device)


        return model



    def _get_optimizer(self, model, optimizer_name, scheduler_dict, lr):
        """get the optimizer and lr_scheduler"""

        if optimizer_name=="Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            raise NotImplementedError('optimizer not implemented') 

        if scheduler_dict['type']=="ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_dict['kwargs'])
        elif scheduler_dict['type']=="MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                            milestones=list(range(50000, 500000, 50000)),
                            **scheduler_dict['kwargs'],)
        else:
            raise NotImplementedError('lr not implemented')

        return optimizer, scheduler



class Denoiser_Trainer_module(trainer.Trainer):
    def __init__(self,
                rank:Union[str, int],
                model:DistributedDataParallel,
                device:Union[str, int],
                lr:optim.lr_scheduler,
                writer_numerical:SummaryWriter,
                writer_imgs:SummaryWriter,
                checkpoints_path:str,
                print_model:bool
            ):

        self.rank = rank
        self.writer_numerical = writer_numerical
        self.model = model

        if self.rank ==0 and print_model:
            model_params = sum(p.numel() for p in self.model.parameters())
            print(f'[info] Denoiser`s params: {model_params}')

        self.device = device
        self.lr = lr

        super(Denoiser_Trainer_module, self).__init__(
                            rank = rank,
                            writer_numerical = writer_numerical, 
                            writer_imgs = writer_imgs,
                            ckpts_path = checkpoints_path
                            ) 

    def _compile(self,
                optimizer:optim,
                scheduler:optim.lr_scheduler,
                loss:Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]=None,
                loss_weights: Dict[str, float]=None,
                metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]={},
                mode='denoiser',
                distributed:bool=False
        )-> None:
        """Configure the denoiser for training.
        Args:
            optimizer: The tf.keras.optimizers optimizer for the model.
            loss: A dictionary with the losses for the model. The sum of the losses is considered.
            loss_weights: Weights of the model losses.
            metrics: evaluation metrics
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.loss_weights = _loss_weights(loss_weights, loss)
        self.mode = mode
        self.metrics = metrics
        self.distributed=distributed


    def _get_train_one_batch(self, data, step_i, data_type=None):
        """training function for one batch"""

        # initilize the total loss value
        loss_tot = torch.tensor(0.0, dtype=torch.float32).cuda(self.device)

        # get ground-truth (gt)
        gt = data['x'].to(torch.float32).cuda(self.device)
        gt_x2, gt_x4 = self._rescale_gt_2d(gt)
        
        # get prediction
        y = data['y'].to(torch.float32).cuda(self.device)
        y_pred, y_pred_x2, y_pred_x4 = self.model(y)


        # apply evaluation metrics
        summary = {}
        for name, fn in self.loss.items():
            loss_  = fn(gt, y_pred) + fn(gt_x2, y_pred_x2) + fn(gt_x4, y_pred_x4) 

            #criterion[name].append(loss_cur)
            loss_tot = loss_tot + loss_ * self.loss_weights[name]
            summary['/loss/' + name] = loss_
        
        summary['/loss/' + 'loss_tot'] = loss_tot

        # write to logger
        if self.rank==0:
            for key, val in summary.items():
                self.writer_numerical.add_scalar(key, val, step_i)

        return summary, loss_tot

    def _rescale_gt_2d(self,im):
        """ downsacle image by factor 2 and 4 """
        im_x2 = F.interpolate(im, size=(im.shape[2]//2, im.shape[3]//2), mode='bilinear',
                    align_corners=False).clamp(min=0, max=1.0)
        im_x4 = F.interpolate(im, size=(im.shape[2]//4, im.shape[3]//4), mode='bilinear',
                    align_corners=False).clamp(min=0, max=1.0)
        return im_x2.to(torch.float32).cuda(self.device), im_x4.to(torch.float32).cuda(self.device)
        

def _loss_weights(loss_weights: Union[None, Dict[str, float]],
                  loss: Dict[str, Any]) -> Dict[str, float]:
    """initilize the weights of loss functions"""

    weights_ = {fn_name: 1. for fn_name in loss}
    if loss_weights is not None:
        weights_.update(loss_weights)
    return weights_
