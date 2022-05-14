from typing import Any, Callable, Dict, Tuple, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data  import DataLoader


from model import trainer
from model import MODELS
from utils.util import load_model

RELU = nn.ReLU(inplace=False)


class Discriminator_Trainer(object):
    def __init__(self,
                disc,
                device,
                in_channels, 
                out_channels,
                writer_numerical, 
                writer_imgs,
                distributed
        )->None:
        r"""
        Args:
            optimizer: .
            los
        """
        super(Discriminator_Trainer, self).__init__() 

        self.device = device
        self.rank =device if distributed else 0
        self.distributed = distributed

        self.epochs = disc.config['epochs']
        self.lr = disc.config['lr']
        self.print_model = disc.config['print_model']

        self.write_path = disc.log_dir
        self.checkpoints_path = disc.ckpt_dir

        self.validation_per_step = disc.config['val_per_step']
        self.test_per_step = disc.config['test_per_step']
        self.ckpts_per_step = disc.config['checkpoint_per_step']

        # get and distribute the model 
        self.model = self._get_model('discriminator', disc.model, 
                            in_channels, out_channels, 
                            self.checkpoints_path,
                            disc.config['negative_slope'])


        # get optimizer
        self.optimizer, self.scheduler = self._get_optimizer(self.model, 
                                            disc.config['optimizer'], 
                                            disc.config['lr_scheduler'], 
                                            disc.config['lr'])

        # get writers
        self.writer_numerical = writer_numerical
        self.writer_imgs = writer_imgs

    
    def __call__(self, 
                ds_train_loader 
                )->bool:

        if self.rank==0:        
            print("[i] Configuring discriminator...")
        trainer = Discriminator_Trainer_module(rank =self.rank, 
                                        model = self.model,
                                        device = self.device,
                                        lr = self.lr,
                                        writer_numerical = self.writer_numerical,
                                        writer_imgs = self.writer_imgs,
                                        checkpoints_path = self.checkpoints_path,
                                        print_model = self.print_model)
        if self.rank==0:
            print("[i] Compiling discriminator...")
        trainer.compile( optimizer = self.optimizer, 
                        scheduler = self.scheduler, 
                        loss = {}, 
                        loss_weights = {}, 
                        metrics = {},
                        distributed=self.distributed)

        if self.rank==0:
            print("[i] Fitting discriminator...")
        trainer.fit(train_data = ds_train_loader, 
                    validation_data = {}, 
                    test_data = {}, 
                    epochs = self.epochs, 
                    validation_per_step = self.validation_per_step,
                    test_per_step = self.test_per_step, 
                    ckpts_per_step = self.ckpts_per_step)

        return True



    def _get_discriminator(self, model_name, in_channels, out_channels, negative_slope):
        """ get discriminator model"""
        module_ = getattr(MODELS, model_name)
        return module_(in_channels, out_channels, negative_slope)

    def _get_model(self, mode, model_name, in_channels, out_channels, ckpt_dir, 
                negative_slope, prefix='ckpts_step_', suffix='.pt'):
        """We get the model and then load its last saved one if avail"""

        # get model
        model = self._get_discriminator(model_name, in_channels, out_channels, negative_slope)

        if self.distributed:
            # map model into DDP
            model = model.cuda(self.device)
            model = DistributedDataParallel(model, device_ids=[self.device]) 
        else:
            model = model.to(self.device)


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


class Discriminator_Trainer_module(trainer.Trainer):
    def __init__(self,
                rank,
                model,
                device,
                lr,
                writer_numerical,
                writer_imgs,
                checkpoints_path,
                print_model):

        self.rank = rank
        self.writer_numerical = writer_numerical
        self.model = model

        if self.rank ==0 and print_model:
            model_params = sum(p.numel() for p in self.model.parameters())
            print(f'[info] Discriminator`s params: {model_params}')

        self.device = device
        self.lr = lr

        super(Discriminator_Trainer_module, self).__init__(
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
                mode='discriminator',
                distributed:bool=False
        )-> None:
        """Configure the discriminator for training.
        Args:
            optimizer: The tf.keras.optimizers optimizer for the model.
            loss: A dictionary with the losses for the model. The sum of the losses is considered.
            loss_weights: Weights of the model losses.
            metrics: evaluation metrics
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.mode = mode
        self.distributed=distributed


    def _get_train_one_batch(self, data, step_i, data_type=None):

        # initilize the total loss value
        loss_ = torch.tensor(0.0, dtype=torch.float32).cuda(self.device)

        if data_type == 'gt':
            # Compute the loss for the true sample
            gt = data['x'].to(torch.float32).cuda(self.device)
            gt_bridge, gt_x0, gt_x2, gt_x4 = self.model(gt) #
            B = gt.shape[0]

            true_ravel = torch.concat([torch.reshape(gt_bridge, [B,-1]),
                                torch.reshape(gt_x0, [B,-1]), 
                                torch.reshape(gt_x2, [B,-1]),
                                torch.reshape(gt_x4, [B,-1])
                                ], axis=-1)
            loss_ = loss_ + torch.mean(RELU(1.0 - true_ravel)) 
        else:
            #prediction
            y = data['y'].to(torch.float32).cuda(self.device)
            y_bridge, y_pred, y_pred_x2, y_pred_x4 = self.model(y)
            B = y.shape[0]

            # Compute the loss for the true sample
            pred_ravel = torch.concat([torch.reshape(y_bridge, [B,-1]),
                                torch.reshape(y_pred, [B,-1]), 
                                torch.reshape(y_pred_x2, [B,-1]),
                                torch.reshape(y_pred_x4, [B,-1])
                                ], axis=-1)
            loss_ = loss_ + torch.mean(RELU(1.0 + pred_ravel)) 


        summary = {}
        summary['/loss/' + 'loss_tot'] = loss_
        # Anything else?!

        if self.rank==0:
            for key, val in summary.items():
                self.writer_numerical.add_scalar(key, val, step_i)

        return summary, loss_