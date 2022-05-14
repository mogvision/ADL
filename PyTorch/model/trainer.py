from typing import Any, Callable, Dict, Tuple, Union
import abc
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
from torch.nn.parallel import DataParallel, DistributedDataParallel

from utils.util import save_model
# save images of validation & test
SAVE_IMGS = True # <<<<<<<<


class Trainer(abc.ABC):
    def __init__(self,
                rank:int,
                ckpts_path: str, 
                writer_numerical: str,
                writer_imgs:str):
        r"""A trainer which can be trained with the API.
        Args:
            rank: current rank
            ckpts_path: Path to the checkpoints file
            writer_numerical: summary file for storing numerical logging
            writer_imgs: summary file for storing the first images of the last batch
        """
        super(Trainer, self).__init__()

        self.rank =rank
        self.writer_numerical = writer_numerical
        self.writer_imgs =  writer_imgs
        self.ckpts_path = ckpts_path

    def compile(self, *args, **kwargs):
        r"""Configures the trainer for training."""
        self._compile(*args, **kwargs)
        self.compiled = True

    def fit(self, 
            train_data, 
            validation_data,
            test_data,
            epochs: int=100,
            validation_per_step: int=1000,
            test_per_step: int=5000, 
            ckpts_per_step: int=5000,
            **kwargs):

        # check whether compiled was excuted ...
        if not self.compiled:
            raise ValueError("Please call compile() before training!")

        # check wether we have test data
        if test_data is None:
            test_data = {}

        decay = 0.

        self.model.train()

        steps = epochs * len(train_data)
        for epoch in range(epochs):
            for i, batch_i in enumerate(train_data):
                step_i = epoch * len(train_data) + i

                ############################################
                #                1/4: Training
                ############################################
                if self.mode == 'denoiser':
                    # train denoiser
                    summary_train, _ = self.train(batch_i, self.optimizer, self.scheduler, None, step_i)


                elif self.mode == 'discriminator':
                    loss_tot = torch.tensor(0.0, dtype=torch.float32).cuda(self.device)

                    # train fakse samples
                    summary_train, _ = self.train(batch_i, self.optimizer, self.scheduler, None, step_i)
                    
                    # train true samples
                    _, loss_ = self.train(batch_i, self.optimizer, self.scheduler, 'gt', step_i)

                    loss_tot += loss_
                    summary_train['/loss/' + 'loss_tot'] = loss_tot


                elif self.mode == 'ADL':
                    loss_tot = torch.tensor(0.0, dtype=torch.float32).cuda(self.device)

                    # Run the discriminator for true sample
                    _, loss_disc = self.train(batch_i, self.optimizer['opt_disc'], 
                                            self.scheduler['sched_disc'], 'gt', step_i)
                    loss_tot += loss_disc

                    # Run the discriminator for fake sample # run the denoiser
                    summary_train, loss_den = self.train(batch_i, self.optimizer['optimizer'], 
                                                    self.scheduler['scheduler'], None, step_i)
                    
                    loss_disc = summary_train['/loss/disc_loss_tot']
                    opt_disc  = self.optimizer['opt_disc']
                    opt_disc.zero_grad()
                    loss_disc.backward()
                    opt_disc.step()
                    loss_tot += loss_disc
                    summary_train['/loss/disc_loss_tot'] = loss_tot


                ############################################
                #                2/4: Validation
                ############################################
                if  (step_i>0) and (step_i%validation_per_step == 0):
                    if  len(validation_data)>0:
                        summary_val = self._get_val(self.model, validation_data, step_i)                        
                        
                    # print out the summary
                    if  self.rank == 0:
                        print(f"\n{datetime.now()}\tstep: <<<<< {step_i}/{steps} >>>>>")
                        self._print_summary('TRAIN', summary_train)
                        if  len(validation_data)>0:
                            self._print_summary('VALIDATION', summary_val)
                
                        # print lr
                        param = 'LR'

                        if self.mode == 'ADL':
                            opt = self.optimizer['optimizer']
                            print(f'{param.ljust(11)}-> {opt.param_groups[0]["lr"]:0.8f}') 
                        else:
                            print(f'{param.ljust(11)}-> {self.optimizer.param_groups[0]["lr"]:0.8f}') 

                ############################################
                #                 3/4: Test
                ############################################
                if (step_i>0) and (step_i%test_per_step == 0) and (len(test_data)>0):
                    summary_test = self._get_test(self.model, test_data, step_i)
                    if  self.rank == 0:
                        self._print_summary('TEST', summary_test)

                ############################################
                #              4/4: Checkpoint
                ############################################
                if (self.rank == 0) and (step_i>0) and (step_i%ckpts_per_step==0):
                    if self.mode == 'ADL':
                        self._save_model(self.model, step_i, self.optimizer['optimizer'])
                    else:
                        self._save_model(self.model, step_i, self.optimizer)
                        print("[i] The checkpoint is stored with id: ", step_i) 


        if self.rank == 0:
            self.writer_numerical.close()
            self.writer_imgs.close()

    def train(self, batch_i, optimizer, scheduler, data_type, step_i):
        summary, loss = self._get_train_one_batch(batch_i, step_i, data_type)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return summary, loss


    def _get_val(self, model, validation_data, step_i):
        model.eval()

        summary = self.metrics.copy()
        for name, fn in self.metrics.items():
            summary[name] = torch.tensor(0.0, dtype=torch.float64).cuda(self.device)
        
        with torch.no_grad():
            for i, batch_i in enumerate(validation_data):

                # get ground-truth
                GT= batch_i['x'].to(torch.float32).cuda(self.device)

                # get noisy data
                y = batch_i['y'].to(torch.float32).cuda(self.device)

                #prediction
                y_pred, _, _ = self.model(y)

                # apply evaluation metrics
                for name, fn in self.metrics.items():
                    val = fn(GT,y_pred)
                    summary[name]+=val
                    if self.rank == 0:
                        self.writer_numerical.add_scalar('VAL/'+name, val, step_i)


            # save the images of the last ietration
            if self.rank == 0 and SAVE_IMGS:
                grid = make_grid(GT)
                self.writer_imgs.add_image('VAL/GT/', grid, 0)

                grid = make_grid(y_pred)
                self.writer_imgs.add_image('VAL/y_pred/', grid, 0)

                # save other scales
                #grid = make_grid(y_pred_x2)
                #self.writer_imgs.add_image('VAL/y_pred_x2/', grid, 0)

                #grid = make_grid(y_pred_x4)
                #self.writer_imgs.add_image('VAL/y_pred_x4/', grid, 0)

                #save_image(y_pred,    f'{self.log_dir_imgs}/val_{step_i}_.png')


        # take average over all results
        for name in self.metrics.keys():
            summary[name] /= (i+1.)
        return summary


    def _get_test(self, model, test_data, step_i):
        model.eval()

        summary = self.metrics.copy()
        for name, fn in self.metrics.items():
            summary[name] = torch.tensor(0.0, dtype=torch.float64).cuda(self.device)
          
        with torch.no_grad():
            for i, batch_i in enumerate(test_data):

                GT = batch_i['x'].to(torch.float32).cuda(self.device)
                y = batch_i['y'].to(torch.float32).cuda(self.device)
        
                #prediction
                y_pred, _, _ = self.model(y)
                    
                for name, fn in self.metrics.items():
                    metric = fn(GT,y_pred)
                    summary[name]+=metric
                    if self.rank==0:
                        self.writer_numerical.add_scalar('TEST/'+name, metric, step_i)

            # save the images of the last ietration
            if self.rank==0 and SAVE_IMGS:
                grid = make_grid(GT)
                self.writer_imgs.add_image('TEST/GT/', grid, 0)

                grid = make_grid(y_pred)
                self.writer_imgs.add_image('TEST/y_pred/', grid, 0)

                # save other scales
                #grid = make_grid(y_pred_x2)
                #self.writer_imgs.add_image('TEST/y_pred_x2/', grid, 0)

                #grid = make_grid(y_pred_x4)
                #self.writer_imgs.add_image('TEST/y_pred_x4/', grid, 0)


        # take average over all results
        for name in self.metrics.keys():
            summary[name] /= (i+1.)
        return summary


    #  auxiliary funcs
    def _save_model(self, model, step_i, optimizer):
        file_ = f"{self.ckpts_path}/ckpts_step_{step_i}.pt"
        save_model(model, file_)
        '''
        torch.save({
                    'step': step_i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, file_)
        '''
        
    def _print_summary(self, prefix, summary):
        precision = '0.5f'
        print(f'{prefix.ljust(11)}-> ', end = '') 
        keys_num = len(summary.keys())

        counter = 1
        for key, value in summary.items():
            key = key.split('/')[-1]
            if counter < keys_num:
                print(f"{key.ljust(4)}: {value:.4f}", end = ' | ') 
            else:
                print(f"{key.ljust(4)}: {value:.4f}", end = ' ')
            counter += 1
        print()
