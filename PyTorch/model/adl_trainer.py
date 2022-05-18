from typing import Any, Callable, Dict, Tuple, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data  import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import trainer
from model import MODELS
from utils.util import load_model

RELU = nn.ReLU(inplace=False)

class ADL_Trainer(object):
    def __init__(self,
                adl:Dict,
                device:Union[str, int],
                in_channels:int, 
                out_channels:int, 
                writer_numerical:SummaryWriter, 
                writer_imgs:SummaryWriter,
                distributed
        )->None:
        super(ADL_Trainer, self).__init__() 


        self.device = device
        self.rank =device if distributed else 0
        self.distributed = distributed

        self.epochs = adl.config['epochs']
        self.lr = adl.config['lr']
        self.print_model =  adl.config['print_model']

        self.metrics = adl.denoiser.eval_params

        self.loss = adl.denoiser.loss
        self.loss_weights = adl.denoiser.loss_weights

        self.write_path = adl.log_dir
        self.checkpoints_path = adl.ckpt_dir

        self.validation_per_step = adl.config['val_per_step']
        self.test_per_step = adl.config['test_per_step']
        self.ckpts_per_step = adl.config['checkpoint_per_step']


        # get and distribute the model 
        self.model, self.model_disc = self._get_model('ADL', 
                                            adl, in_channels, out_channels, 
                                            self.checkpoints_path)

        # get optimizer
        self.optimizer, self.scheduler = self._get_optimizer(self.model, 
                                            adl.config['optimizer'], 
                                            adl.config['lr_scheduler'], 
                                            adl.config['lr'])

        # get writers
        self.writer_numerical = writer_numerical
        self.writer_imgs = writer_imgs

    
    def __call__(self, 
                ds_train_loader, 
                ds_valid_loader, 
                ds_test_loader 
                )->bool:

        if self.rank==0:
            print("[i] Configuring ADL...")
        trainer = ADL_Trainer_module(rank =self.rank, 
                                        model_denoiser = self.model,
                                        model_disc = self.model_disc,
                                        device = self.device,
                                        lr = self.lr,
                                        writer_numerical = self.writer_numerical,
                                        writer_imgs = self.writer_imgs,
                                        checkpoints_path = self.checkpoints_path,
                                        print_model = self.print_model)
        if self.rank==0:
            print("[i] Compiling ADL...")
        trainer.compile( optimizer = self.optimizer, 
                        scheduler = self.scheduler, 
                        loss = self.loss, 
                        loss_weights = self.loss_weights, 
                        metrics = self.metrics,
                        distributed=self.distributed)

        if self.rank==0:
            print("[i] Fitting ADL...")
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

    def _get_discriminator(self, model_name, in_channels, out_channels, negative_slope):
        """ get discriminator model"""
        module_ = getattr(MODELS, model_name)
        return module_(in_channels, out_channels, negative_slope)


    def _get_model(self, mode, adl, 
                    in_channels, out_channels, 
                    ckpt_dir):
        """We get the model and then load its last saved one if avail"""

        # get models
        model = self._get_denoiser(adl.denoiser.model, in_channels, out_channels)
        model_disc = self._get_discriminator(adl.disc.model, in_channels, 
            out_channels, adl.disc.config['negative_slope'])


        if self.distributed:
            # map model into DDP
            model = model.cuda(self.device)
            model = DistributedDataParallel(model, device_ids=[self.device]) 

            model_disc = model_disc.cuda(self.device)
            model_disc = DistributedDataParallel(model_disc, device_ids=[self.device]) 
        else:
            model = model.to(self.device)
            model_disc = model_disc.to(self.device)

        # load the saved denoiser model if avail
        model = self._restore_model('ADL:denoiser', adl.denoiser.ckpt_dir, model, self.device)
        model_disc = self._restore_model('ADL:disc',adl.disc.ckpt_dir, model_disc, self.device)

        # load the adl denoiser if avail
        model = self._restore_model('ADL',adl.ckpt_dir, model, self.device)

        return model, model_disc


    def _restore_model(self, mode, ckpt_dir, model, device, prefix='ckpts_step_', suffix='.pt'):
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

        return model.to(device)



    def _get_optimizer(self, model, optimizer_name, scheduler_dict, lr):
        if optimizer_name=="Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            raise NotImplementedError 


        if scheduler_dict['type']=="ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_dict['kwargs'])
        elif scheduler_dict['type']=="MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                            milestones=list(range(50000, 500000, 50000)),
                            **scheduler_dict['kwargs'],)
        else:
            raise NotImplementedError('lr not implemented')

        return {
                'optimizer': optimizer, 
                'opt_disc': optimizer
            }, {
                'scheduler':scheduler, 
                'sched_disc':scheduler
            }



class ADL_Trainer_module(trainer.Trainer):
    def __init__(self,
                rank:Union[str, int],
                model_denoiser:DistributedDataParallel,
                model_disc:DistributedDataParallel,
                device:Union[str, int],
                lr:optim.lr_scheduler,
                writer_numerical:SummaryWriter,
                writer_imgs:SummaryWriter,
                checkpoints_path:str,
                print_model:bool
                ):

        self.rank = rank
        self.writer_numerical = writer_numerical
        self.model = model_denoiser
        self.model_disc = model_disc

        if self.rank ==0 and print_model: 
            model_params = sum(p.numel() for p in self.model.parameters())
            print(f'[info] Denoiser`s params: {model_params}')

            model_params = sum(p.numel() for p in self.model_disc.parameters())
            print(f'[info] Discriminator`s params: {model_params}')

        self.device = device
        self.lr = lr

        super(ADL_Trainer_module, self).__init__(
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
                mode='ADL',
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
        """train the denoiser for one batch"""
        summary = {}

        if data_type == 'gt':
            loss = torch.tensor(0.0, dtype=torch.float32).cuda(self.device)

            # Run the discriminator ------------------------
            # Compute the loss for the true sample
            gt = data['x'].to(torch.float32).cuda(self.device)
            gt_bridge, gt_x0, gt_x2, gt_x4 = self.model_disc(gt)
            B = gt.shape[0]

            true_ravel = torch.concat([torch.reshape(gt_bridge, [B,-1]),
                                torch.reshape(gt_x0, [B,-1]), 
                                torch.reshape(gt_x2, [B,-1]),
                                torch.reshape(gt_x4, [B,-1])
                                ], axis=-1)
            loss_disc = torch.mean(RELU(1.0 - true_ravel)) 
            summary['/loss/' + 'disc_loss_tot'] = loss_disc
            loss = loss_disc



        else:
            # Run the denoiser model -------------------------            
            loss = torch.tensor(0.0, dtype=torch.float32).cuda(self.device)
            gt = data['x'].to(torch.float32).cuda(self.device)
            gt_x2, gt_x4 = self._rescale_gt_2d(gt)

            #prediction
            y = data['y'].to(torch.float32).cuda(self.device)
            y_pred, y_pred_x2, y_pred_x4 = self.model(y)
            B = y.shape[0]

            for name, fn in self.loss.items():
                loss_ = fn(gt, y_pred) + fn(gt_x2, y_pred_x2) + fn(gt_x4, y_pred_x4) 

                #criterion[name].append(loss_cur)
                loss = loss + loss_ * self.loss_weights[name]
                summary['/loss/' + name] = loss_
            summary['/loss/' + 'loss_tot'] = loss

            # Run the discriminator -------------------------
            y_bridge, y_pred, y_pred_x2, y_pred_x4 = self.model_disc(y)

            # Compute the loss for the true sample
            pred_ravel = torch.concat([torch.reshape(y_bridge, [B,-1]),
                                torch.reshape(y_pred, [B,-1]), 
                                torch.reshape(y_pred_x2, [B,-1]),
                                torch.reshape(y_pred_x4, [B,-1])
                                ], axis=-1)
            loss_disc =  torch.mean(RELU(1.0 + pred_ravel))
            summary['/loss/' + 'disc_loss_tot'] = loss_disc

        if self.rank==0:
            for key, val in summary.items():
                self.writer_numerical.add_scalar(key, val, step_i)
        return summary, loss

    def _rescale_gt_2d(self,im):
        """ downsacle image by factor 2 and 4 """
        im_x2 = F.interpolate(im, size=(im.shape[2]//2, im.shape[3]//2), mode='bilinear',
                    align_corners=False).clamp(min=0, max=1.0)
        im_x4 = F.interpolate(im, size=(im.shape[2]//4, im.shape[3]//4), mode='bilinear',
                    align_corners=False).clamp(min=0, max=1.0)
        return im_x2.to(torch.float32).cuda(self.device), im_x4.to(torch.float32).cuda(self.device)


def _loss_weights(loss_weights: Union[None, Dict[str, float]],
                  loss: Dict[str, Any]) -> Dict[str, float]:
    weights_ = {fn_name: 1. for fn_name in loss}
    if loss_weights is not None:
        weights_.update(loss_weights)
    return weights_
