import os
import tensorflow as tf
from tensorflow import keras
from typing import Any, Callable, Dict, Tuple, Union

from util import utils
from model import trainer as base_trainer
from model import lr as lr_py
from model import models

tf.config.experimental.enable_tensor_float_32_execution(True)
#tf.config.experimental.set_synchronous_execution(False)


class ADL_Trainer(object):
    def __init__(self, 
                strategy,
                gpus_num,
                params_struct_adl=None,
                params_struct_denoiser=None,
                params_struct_disc=None,
                args = None,
                channels_num:int=-1,
                steps_per_epoch:Dict[str,int]=None,
                load_save_device:str=None
                ):
        super(ADL_Trainer, self).__init__()

        self.strategy = strategy
        self.gpus_num = gpus_num
        self.config = params_struct_adl.config
        self.params = params_struct_adl
        self.args = args
        self.channels_num = channels_num
        self.steps_per_epoch = steps_per_epoch

        self.params_denoiser = params_struct_denoiser
        self.config_denoiser = params_struct_denoiser.config
        self.config_disc = params_struct_disc.config
        self.params_disc = params_struct_disc

        self.params_disc.loss = None
        self.params_disc.loss_weights = None

        # determice the device for loading and saving the model
        self.options = utils.struct_cls()
        self.options.save_options = tf.saved_model.SaveOptions(experimental_io_device=load_save_device) 
        self.options.load_options = tf.saved_model.LoadOptions(experimental_io_device=load_save_device)

        # Create the denoiser, optimizer and metrics inside the strategy scope, so that the
        # variables can be mirrored on each device.
        with strategy.scope():
            denoiser = self._get_denoiser(self.config_denoiser["model"], channels_num, channels_num)
            disc = self._get_discriminator(self.config_disc["model"], channels_num)

            
            # load denoiser if avail
            ckpt_dir =  os.path.join(self.params.results_dir, 'ADL', 'checkpoints')
            self.denoiser, status = self._get_model(denoiser, ckpt_dir, self.options, 'denoiser')

            if status:
                ckpt_dir =  os.path.join(self.params.results_dir, 'denoiser', 'checkpoints')
                self.denoiser, status = self._get_model(denoiser, ckpt_dir, self.options, 'denoiser')


            # load discriminator if avail
            ckpt_dir =  os.path.join(self.params.results_dir, 'discriminator', 'checkpoints')
            self.disc, _ = self._get_model(disc, ckpt_dir, self.options, 'discriminator')

            # get the LR and optimizer
            self.optimizer_ = self._get_optimiers(self.config_denoiser['lr'])
            self.lr_=self._get_lr(self.config_denoiser['lr_scheduler'])

        # remove the log and checkpoint dirs if running from initial_ckpt == 0
        if False: #self.initial_ckpt < 1:
            utils.rmtree_dirs(self.params.ckpt_dir)
            utils.rmtree_dirs(self.params.log_dir)


        ## print model summary
        if self.config['print_model']:
            utils.get_model_summary(self.denoiser, f"ADL_{self.config_denoiser['model']}")
            utils.get_model_summary(self.disc, f"ADL_{self.config_disc['model']}")

    def __call__(self, ds_train, ds_val, ds_test):
        tf.print("[i] Compiling ADL...")
        global_batch_size = self.params.bs_per_gpu*self.gpus_num

        assert tf.distribute.get_replica_context() is not None 

        with self.strategy.scope():
            # Configure generator  =============
            trainer = ADL_Trainer_module(model_denoiser=self.denoiser, 
                                        model_disc=self.disc,
                                        ckpts_file=self.params.ckpt_dir, 
                                        log_dir=self.params.log_dir,
                                        strategy=self.strategy, 
                                        gpus_num=self.gpus_num, 
                                        options=self.options, 
                                        steps_per_epoch=self.steps_per_epoch, 
                                        checkpoints_per_step=self.config['checkpoint_per_step'],
                                        global_batch_size=global_batch_size,
                                        disc_per_step=self.config['disc_per_step'])

            trainer.compile( optimizer_denoiser=self.optimizer_,
                            optimizer_disc=self._get_optimiers(self.config['lr']),
                            loss_denoiser=self.params_denoiser.loss,
                            loss_disc=self.params_disc.loss,
                            loss_weights_denoiser=self.params_denoiser.loss_weights,
                            loss_weights_disc=self.params_disc.loss_weights,
                            metrics=self.params_denoiser.eval_params,
                            lr=self.lr_)

            steps = self.config['epochs'] * self.steps_per_epoch['TRAIN']

            trainer.fit(train_data=ds_train, 
                        validation_data=ds_val,
                        test_data=ds_test,
                        steps=steps, 
                        initial_step = self.config['checkpoint_per_step'],
                        validation_per_step=self.config['val_per_step'],
                        test_per_step=self.config['test_per_step'],
                        checkpoints_per_step=self.config['checkpoint_per_step'])
        return True


    def _get_denoiser(self, model_name, in_channels, out_channels)->keras.Model:
        denosier = getattr(models, model_name)
        return denosier(in_channels, out_channels)

    def _get_discriminator(self, model_name, channels_num)->keras.Model:
        discriminator = getattr(models, model_name)
        return discriminator(channels_num)

    def _get_model(self, model, ckpt_dir, options, mode, prefix='checkpoint-')->keras.Model:
        model_path = None
        if os.path.exists(ckpt_dir):
            step_ids = [int(dir_.split(prefix)[-1]) for dir_ in os.listdir(ckpt_dir)]
            if len(step_ids) > 0:
                step_num = max(step_ids)
                model_path = f"{ckpt_dir}/{prefix}{step_num}/"

        if model_path:
            loaded_model = tf.keras.models.load_model(model_path, options=options.load_options)
            tf.print(f"[i] {mode}: Restoring the model from step {step_num}.")
            return loaded_model, False
        
        tf.print(f"[i] {mode}: Creating a new model.")
        return model, True
    

    def _get_optimiers(self, lr):
        return tf.keras.optimizers.Adam(lr)

    def _get_lr(self, config):
        if config['type'] == 'ReduceLROnPlateau':
            return lr_py.ReduceLROnPlateau(optimizer=self.optimizer_, **config['kwargs'])
        else:
            tf.print("The type of requested LR is empty or not implemented yet!")
            return None



class ADL_Trainer_module(base_trainer.Multiple_Steps_Trainer):
    def __init__(self, 
            model_denoiser: keras.Model, 
            model_disc: keras.Model, 
            ckpts_file: str=None, 
            log_dir: str=None,
            strategy=None,
            gpus_num:int=-1,
            options=None, 
            steps_per_epoch:Dict[str,int]=None,
            checkpoints_per_step:int=-1,
            global_batch_size:int=None, 
            disc_per_step:int=None) -> None:
        r"""Denoiser_Trainer_module is ....
        Args:
            model: The model
            ckpts_file: checkpoints file
            log_dir : directory of log-file
        """

        self.model = model_denoiser
        self.model_disc = model_disc
        self.HW = tf.cast([-1,-1], dtype=tf.int32) # convert to tensor
        self.gpus_num = gpus_num
        self.checkpoints_per_step = checkpoints_per_step
        self.global_batch_size = global_batch_size
        self.disc_per_step = disc_per_step
        
        # save valid/test results
        self.return_image = True # If True, the first image of valid/test is stored

        if gpus_num>1:
            # we make `return_image` FALSE for More than one GPU
            self.return_image = False 


        super(ADL_Trainer_module, self).__init__(
                            ckpts_file=ckpts_file, 
                            log_dir=log_dir, 
                            strategy=strategy,
                            gpus_num=gpus_num,
                            options=options,
                            global_batch_size=self.global_batch_size, 
                            steps_per_epoch=steps_per_epoch) 

    def _compile(self,
                 optimizer_denoiser: keras.optimizers.Optimizer,
                 optimizer_disc: keras.optimizers.Optimizer,
                 loss_denoiser: Dict[str, Callable[[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]],
                 loss_disc: Dict[str, Callable[[tf.Tensor, tf.Tensor], tf.Tensor]],
                 loss_weights_denoiser: Dict[str, float] = None,
                 loss_weights_disc: Dict[str, float] = None,
                 metrics: Dict[str, Callable[[tf.Tensor, tf.Tensor], tf.Tensor]] = None,
                 lr=None) -> None:
        r"""Configure ADL for training.
        Args:
            optimizer_denoiser: The tf.keras.optimizers optimizer for the denoiser.
            optimizer_disc: The tf.keras.optimizers optimizer for the discriminator.
            loss_denoiser: A dictionary of  losses for the denoiser.
            loss_disc: A dictionary of  losses for the discriminator.
            loss_weights_denoiser: Weights of the denoiser losses.
            loss_weights_disc: Weights of the discriminator losses.
            metrics: evaluation metrics
            lr: learning rate
        """
        self.optimizer_denoiser = optimizer_denoiser
        self.optimizer_disc = optimizer_disc
        self.loss_denoiser = loss_denoiser
        self.loss_disc = loss_disc
        self.loss_weights_denoiser = _loss_weights(loss_weights_denoiser, loss_denoiser)
        self.loss_weights_disc = _loss_weights(loss_weights_disc, loss_disc)
        self.net = 'ADL'
        self.lr = lr
        self.metrics = metrics #evaluation metrics
        self.relu=tf.keras.activations.relu

    #@tf.function
    def train_one_step(self, data:tf.data.Dataset, step:tf.Variable) -> Dict[str, tf.Tensor]:
        r"""Run the denoiser for one step.
        Args:
            data: the structure of data is as follows: 
                    y, x_gt, label = data
                    y: noisy image
                    x_gt: ground-truth clean image
                    label: 0 (noisy) or 1 (clean)
            loss: A dictionary with the losses for the model. The sum of the losses is considered.
            loss_weights: Weights of the model losses.
            metrics: evaluation metrics
        """

        # get data
        y, x_gt, label = data
        B = x_gt.get_shape()[0]

        # create pyramid from ground-truth data. In this study, our pyramid has 
        # three scales inclduing x0 (input image), x2 (half-image) and x4 (quadratic image)
        # >>>> NOTE that `2x is meant 2 times` and `x2 is meant  1/2` <<<<
        x_gt_x2 = tf.keras.layers.AveragePooling2D(pool_size=2)(x_gt)
        x_gt_x4 = tf.keras.layers.AveragePooling2D(pool_size=4)(x_gt)

        # initilize summary 
        summary = {}
        loss_dict_denoiser = {}
        #loss_dict_disc = {}

        [loss_dict_denoiser.update({key: tf.constant(0, tf.float32)}) for key in self.loss_denoiser.keys()]
        #[loss_dict_disc.update({key: tf.constant(0, tf.float32)}) for key in self.loss_disc.keys()]

        loss_denoiser = tf.constant(0, tf.float32)
        loss_disc = tf.constant(0, tf.float32)

        with tf.GradientTape() as tape_denoiser, tf.GradientTape() as tape_disc:

            # Run the denoiser model -----------------------------
            yhat, yhat_x2, yhat_x4 = self.model(y, training=True)

            # Compute the loss
            for loss_id, loss_fn in self.loss_denoiser.items():
                # compute loss of each scale and then sum up
                loss_dict_denoiser[loss_id] += loss_fn(x_gt,    yhat   , self.global_batch_size) # at scale 0
                loss_dict_denoiser[loss_id] += loss_fn(x_gt_x2, yhat_x2, self.global_batch_size) # at scale x2
                loss_dict_denoiser[loss_id] += loss_fn(x_gt_x4, yhat_x4, self.global_batch_size) # at scale x4

            # apply weights    
            for loss_id, loss_ in loss_dict_denoiser.items():    
                loss_denoiser += self.loss_weights_denoiser[loss_id] * loss_ 
                summary['TRAIN/'+loss_id] = loss_
                #tf.print(step, loss_id, loss_)


            # Run the discriminator model -----------------------------
            bridge_map_true, yhat_x0_map_true, yhat_x2_map_true, yhat_x4_map_true = self.model_disc(x_gt, training=True)            

            # Compute the loss for the true sample
            true_ravel = tf.concat([tf.reshape(yhat_x0_map_true, [B,-1]), 
                                    tf.reshape(yhat_x2_map_true, [B,-1]),
                                    tf.reshape(yhat_x4_map_true, [B,-1])], 
                                    axis=-1)

            loss_disc += tf.reduce_mean(self.relu(1.0 - true_ravel)) 
            loss_disc += tf.reduce_mean(self.relu(1.0 - tf.reshape(bridge_map_true, [B,-1]))) 

            # Compute the loss for the fake sample
            bridge_map_pred, yhat_x0_map_pred, yhat_x2_map_pred, yhat_x4_map_pred = self.model_disc(y, training=True)
            pred_ravel = tf.concat([tf.reshape(yhat_x0_map_pred, [B,-1]), 
                                    tf.reshape(yhat_x2_map_pred, [B,-1]),
                                    tf.reshape(yhat_x4_map_pred, [B,-1])], 
                                    axis=-1)

            loss_disc += tf.reduce_mean(self.relu(1.0 + pred_ravel)) 
            loss_disc += tf.reduce_mean(self.relu(1.0 + tf.reshape(bridge_map_pred, [B,-1]))) 

        # log train results
        summary['TRAIN/denosier_tot'] = loss_denoiser
        summary['TRAIN/disc_loss'] = loss_disc


        # Get the gradients
        grad_denoiser = tape_denoiser.gradient(loss_denoiser, self.model.trainable_variables)
        grad_disc = tape_disc.gradient(loss_disc, self.model_disc.trainable_variables)


        # Apply the gradinets
        self.optimizer_denoiser.apply_gradients(zip(grad_denoiser, self.model.trainable_variables))
        if step%self.disc_per_step == 0:
            self.optimizer_disc.apply_gradients(zip(grad_disc, self.model_disc.trainable_variables))

        return summary




    #@tf.function
    def val_test_one_step(self, 
                data:tf.data.Dataset, 
                step:tf.Variable,
                prefix: str) : #-> [Dict[str, tf.Tensor], Dict[str, tf.Tensor]]
        r"""Run the denoiser for one step on val/test dataset.
        Args:
            data: the structure of data is as follows: 
                    y, x_gt, label = data
                    y: noisy image
                    x_gt: ground-truth clean image
                    label: 0 (noisy) or 1 (clean)
            step = the current step
            prefix: A prefix for the summary.
        """
        summary, img_summary = {}, {}
        for param, metric in self.metrics.items():
            summary[f"{prefix}/{param}"] = tf.constant(0., tf.float32)

        # get data
        y, x_gt, _= data
        B, H, W, C = x_gt.get_shape()

        if tf.rank(y) < 4: 
            return summary, img_summary

        # get prediction
        if prefix == 'TEST':
            model_pred, model_pred_x2, model_pred_x4 = self.model(y, training=False)
        else:
            model_pred, model_pred_x2, model_pred_x4 = self.model(y, training=True)


        if self.return_image:
            for param in ['y', 'x_gt']:     
                img_summary[f"{prefix}/{param}"] = tf.zeros(shape=(1, H, W, C), dtype=tf.float32) 
            ratio = 1
            for param in ['pred', 'pred_x2', 'pred_x4']:     
                img_summary[f"{prefix}/{param}"] = tf.zeros(shape=(1, H//ratio, W//ratio, C), dtype=tf.float32) 
                ratio *= 2
                
        # save the first image of each batch if return_image is True 
        if self.return_image:  
            tmp = model_pred[0,...] 
            img_summary[f"{prefix}/pred"] = tf.expand_dims(tmp, axis=0) 

            tmp = model_pred_x2[0,...] 
            img_summary[f"{prefix}/pred_x2"] = tf.expand_dims(tmp, axis=0) 

            tmp = model_pred_x4[0,...] 
            img_summary[f"{prefix}/pred_x4"] = tf.expand_dims(tmp, axis=0) 

            tmp = y[0,...]  #tf.nn.compute_average_loss(model_pred, global_batch_size=self.global_batch_size)  
            img_summary[f"{prefix}/y"] = tf.expand_dims(tmp, axis=0)  

            tmp = x_gt[0,...]  #tf.nn.compute_average_loss(model_pred, global_batch_size=self.global_batch_size)  
            img_summary[f"{prefix}/x_gt"] = tf.expand_dims(tmp, axis=0) 


        # apply the metrics on the given results
        for param, metric in self.metrics.items():
            val = metric(x_gt, model_pred, self.global_batch_size)
            summary[f"{prefix}/{param}"] = tf.constant(val, tf.float32)

        return summary, img_summary

def _loss_weights(loss_weights: Union[None, Dict[str, float]],
                  loss: Dict[str, Any]) -> Dict[str, float]:

    if loss is None:
        return None
    weights_ = {fn_name: 1. for fn_name in loss}
    if loss_weights is not None:
        weights_.update(loss_weights)
    return weights_



