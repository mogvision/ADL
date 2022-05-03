import os
import tensorflow as tf
from tensorflow import keras
from typing import Any, Callable, Dict, Tuple, Union

from util import utils
from model import trainer as base_trainer
from model import lr as lr_py
from model import models

import util.DataLoader as DataLoader

tf.config.experimental.enable_tensor_float_32_execution(True)
#tf.config.experimental.set_synchronous_execution(False)


class Discriminator_Trainer(object):
    def __init__(self, 
                strategy,
                gpus_num,
                params_struct=None,
                args = None,
                channels_num=None,
                steps_per_epoch:Dict[str,int]=None,
                load_save_device:str=None
                ):
        super(Discriminator_Trainer, self).__init__()

        self.strategy = strategy
        self.gpus_num = gpus_num
        self.config = params_struct.config
        self.params = params_struct
        self.args = args
        self.channels_num = channels_num
        self.steps_per_epoch = steps_per_epoch


        # determice the device for loading and saving the model
        self.options = utils.struct_cls()
        self.options.save_options = tf.saved_model.SaveOptions(experimental_io_device=load_save_device) 
        self.options.load_options = tf.saved_model.LoadOptions(experimental_io_device=load_save_device)


        # Create the discriminator, optimizer and metrics inside the strategy scope, so that the
        # variables can be mirrored on each device.
        with strategy.scope():
            self.disc = self._get_model(self.config['model'], 
                                        self.params.ckpt_dir, 
                                        self.params.log_dir,
                                        self.options, 
                                        'discriminator')

            self.optimizer_ = self._get_optimiers(self.config['lr'])


        # print model summary
        if self.config['print_model']:
            utils.get_model_summary(self.disc, f"ADL_{self.config['model']}")


    def __call__(self, ds_train):

        tf.print("[i] Compiling discriminator...")
        global_batch_size = self.params.bs_per_gpu*self.gpus_num

        assert tf.distribute.get_replica_context() is not None 

        with self.strategy.scope():
            # Configure discriminator  =============
            trainer = Discriminator_Trainer_module(model=self.disc, 
                                ckpts_file=self.params.ckpt_dir, 
                                log_dir=self.params.log_dir,
                                strategy=self.strategy, 
                                gpus_num=self.gpus_num, 
                                options=self.options, 
                                steps_per_epoch=self.steps_per_epoch, 
                                global_batch_size=global_batch_size)

            lr_ = self._get_lr(self.config['lr_scheduler'])

            trainer.compile( optimizer = self.optimizer_, 
                        loss = None,  
                        loss_weights = None, 
                        metrics = None,
                        label_noisy_probability=self.config['label_noisy_probability'],
                        lr=lr_)

            steps = self.config['epochs'] * self.steps_per_epoch['TRAIN']
            trainer.fit(train_data = ds_train, 
                        validation_data = None,
                        test_data = None,
                        steps = steps, 
                        initial_step = self.config['checkpoint_per_step'],
                        validation_per_step = self.config['val_per_step'],
                        test_per_step = self.config['test_per_step'],
                        checkpoints_per_step = self.config['checkpoint_per_step'])
        return True

    def _get_discriminator(self, model_name, channels_num)->keras.Model:
        discriminator = getattr(models, model_name)
        return discriminator(channels_num)

    def _get_optimiers(self, lr):
        return tf.keras.optimizers.Adam(lr)

    def _get_lr(self, config):
        if config['type'] == 'ReduceLROnPlateau':
            return lr_py.ReduceLROnPlateau(optimizer=self.optimizer_, **config['kwargs'])
        else:
            tf.print("The type of requested LR is empty or not implemented yet!")
        return None

    def _get_model(self, model, ckpt_dir, log_dir, options, mode, prefix='checkpoint-')->keras.Model:
        model_path = None
        if os.path.exists(ckpt_dir):
            step_ids = [int(dir_.split(prefix)[-1]) for dir_ in os.listdir(ckpt_dir)]
            if len(step_ids) > 0:
                step_num = max(step_ids)
                model_path = f"{ckpt_dir}/{prefix}{step_num}/"

        if model_path:
            loaded_model = tf.keras.models.load_model(model_path, options=options.load_options)
            tf.print(f"[i] {mode}: Restoring the model from step {step_num}.")
            return loaded_model

        tf.print(f"\n[i] {mode}: Creating a new model.")
        model = self._get_discriminator(model, self.channels_num)
        
        # remove the log and checkpoint dirs if running from the begining
        utils.rmtree_dirs(ckpt_dir)
        utils.rmtree_dirs(log_dir)
        return model




class Discriminator_Trainer_module(base_trainer.Multiple_Steps_Trainer):
    def __init__(self, 
            model:keras.Model, 
            ckpts_file:str=None, 
            log_dir:str=None,
            strategy=None,
            gpus_num:int=-1,
            options=None, 
            steps_per_epoch:Dict[str,int]=None,
            global_batch_size:int=None) -> None:
        """Discriminator_Trainer_module is ....
        Args:
            model: The model
            ckpts_file: checkpoints file
            log_dir : directory of log-file
        """

        self.model = model
        self.HW = tf.cast([-1,-1], dtype=tf.int32) # convert to tensor
        self.gpus_num=gpus_num
        self.global_batch_size = global_batch_size

        super(Discriminator_Trainer_module, self).__init__(
                            ckpts_file=ckpts_file, 
                            log_dir=log_dir, 
                            strategy=strategy,
                            gpus_num=gpus_num,
                            options=options,
                            global_batch_size=self.global_batch_size, 
                            steps_per_epoch=steps_per_epoch) 

    def _compile(self,
                 optimizer:keras.optimizers.Optimizer,
                 loss:Dict[str, Callable[[tf.Tensor, tf.Tensor], tf.Tensor]],
                 loss_weights: Dict[str, float] = None,
                 metrics:Dict[str, Callable[[tf.Tensor, tf.Tensor], tf.Tensor]] = None,
                 label_noisy_probability:float=0.5,
                 lr=None) -> None:
        """Configure the discriminator for training.
        Args:
            optimizer: The tf.keras.optimizers optimizer for the model.
            loss: A dictionary with the losses for the model. The sum of the losses is considered.
            loss_weights: Weights of the model losses.
            metrics: evaluation metrics
            label_noisy_probability: 
            lr: learning rate
        """
        self.optimizer = optimizer
        self.net = 'discriminator'
        self.label_noisy_probability=label_noisy_probability
        self.lr = lr
        self.relu=tf.keras.activations.relu

        #metrics: The metrics that should be evaluated.
        if metrics is not None: 
            self.metrics = metrics
        else:
            self.metrics = {}


    #@tf.function
    def train_one_step(self, data:tf.data.Dataset, step:tf.Variable) -> Dict[str, tf.Tensor]:
        """Run the discriminator for one step.
        Args:
            data: the structure of data is as follows: 
                    y, x_gt, label = data
                    y: noisy image
                    x_gt: ground-truth clean image
                    label: 0 (noisy) or 1 (clean)
            metrics: evaluation metrics
        """

        # get data
        y, x_gt, label = data
        B = x_gt.get_shape()[0]

        # initilize summary 
        summary = {}
        loss = tf.constant(0, tf.float32)

        # Tape the gradients
        with tf.GradientTape() as grad_tape:

            # create pyramid from ground-truth data. In this study, our pyramid has 
            # three scales inclduing x0 (input image), x2 (half-image) and x4 (quadratic image)
            # NOTE that `2x is meant 2 times` and `x2 is meant  1/2` <<<<
            bridge_map_true, yhat_x0_map_true, yhat_x2_map_true, yhat_x4_map_true = self.model(x_gt, training=True)            

            # Compute the loss for the true sample
            loss += tf.reduce_mean(self.relu(1.0 - bridge_map_true)) 
            loss += tf.reduce_mean(self.relu(1.0 - yhat_x0_map_true)) 
            loss += tf.reduce_mean(self.relu(1.0 - yhat_x2_map_true)) 
            loss += tf.reduce_mean(self.relu(1.0 - yhat_x4_map_true)) 


            # Compute the loss for the fake sample
            bridge_map_pred, yhat_x0_map_pred, yhat_x2_map_pred, yhat_x4_map_pred = self.model(y, training=True)
            loss += tf.reduce_mean(self.relu(1.0 + bridge_map_pred)) 
            loss += tf.reduce_mean(self.relu(1.0 + yhat_x0_map_pred)) 
            loss += tf.reduce_mean(self.relu(1.0 + yhat_x2_map_pred)) 
            loss += tf.reduce_mean(self.relu(1.0 + yhat_x4_map_pred)) 
            
        # Return the total loss
        summary['TRAIN/loss'] = loss

        # Get the gradients
        grad = grad_tape.gradient(loss, self.model.trainable_variables)

        # Apply the gradinets
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        return summary



def _loss_weights(loss_weights: Union[None, Dict[str, float]],
                  loss: Dict[str, Any]) -> Dict[str, float]:
    weights_ = {fn_name: 1. for fn_name in loss}
    if loss_weights is not None:
        weights_.update(loss_weights)
    return weights_

