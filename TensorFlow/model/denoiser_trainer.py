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


class Denoiser_Trainer(object):
    def __init__(self, 
                strategy,
                gpus_num,
                params_struct=None,
                args = None,
                channels_num:int=-1,
                steps_per_epoch:Dict[str,int]=None,
                load_save_device:str=None
                ):
        super(Denoiser_Trainer, self).__init__()

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

        # Create the denoiser, optimizer and metrics inside the strategy scope, so that the
        # variables can be mirrored on each device.
        with strategy.scope():
            self.denoiser = self._get_model(self.config['model'], 
                                                self.params.ckpt_dir, 
                                                self.params.log_dir,
                                                self.options, 
                                                'denoiser')
            self.optimizer_ = self._get_optimiers(self.config['lr'])


        # print model summary
        if self.config['print_model']:
            utils.get_model_summary(self.denoiser, f"ADL_{self.config['model']}")


    def __call__(self, ds_train, ds_val, ds_test):
        tf.print("[i] Compiling denoiser...")
        global_batch_size = self.params.bs_per_gpu*self.gpus_num


        assert tf.distribute.get_replica_context() is not None 

        if True: #with self.strategy.scope():
            # Configure generator  =============
            trainer = Denoiser_Trainer_module(model=self.denoiser, 
                                ckpts_file=self.params.ckpt_dir, 
                                log_dir=self.params.log_dir,
                                strategy=self.strategy, 
                                gpus_num=self.gpus_num, 
                                options=self.options, 
                                steps_per_epoch=self.steps_per_epoch, 
                                global_batch_size=global_batch_size)

            lr_ = self._get_lr(self.config['lr_scheduler'])

            trainer.compile( optimizer = self.optimizer_, 
                        loss = self.params.loss,  
                        loss_weights = self.params.loss_weights, 
                        metrics = self.params.eval_params,
                        lr=lr_)

            steps = self.config['epochs'] * self.steps_per_epoch['TRAIN']

            trainer.fit(train_data = ds_train, 
                        validation_data = ds_val,
                        test_data = ds_test,
                        steps = steps,
                        initial_step =  self.config['checkpoint_per_step'],
                        validation_per_step = self.config['val_per_step'],
                        test_per_step = self.config['test_per_step'],
                        checkpoints_per_step = self.config['checkpoint_per_step'])
        return True


    def _get_optimiers(self, lr):
        return tf.keras.optimizers.Adam(lr)


    def _get_lr(self, config):
        if config['type'] == 'ReduceLROnPlateau':
            return lr_py.ReduceLROnPlateau(optimizer=self.optimizer_, **config['kwargs'])
        else:
            tf.print("The type of requested LR is empty or not implemented yet!")
            return None


    def _get_denoiser(self, model_name, in_channels, out_channels)->keras.Model:
        denosier = getattr(models, model_name)
        return denosier(in_channels, out_channels)


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
        
        tf.print(f"[i] {mode}: Creating a new model.")
        model = self._get_denoiser(model, self.channels_num, self.channels_num)
        
        # remove the log and checkpoint dirs if running from the begining
        utils.rmtree_dirs(ckpt_dir)
        utils.rmtree_dirs(log_dir)
        return model



class Denoiser_Trainer_module(base_trainer.Multiple_Steps_Trainer):
    def __init__(self, 
            model: keras.Model, 
            ckpts_file: str=None, 
            log_dir: str=None,
            strategy=None,
            gpus_num:int=-1,
            options=None, 
            steps_per_epoch:Dict[str,int]=None,
            global_batch_size:int=None) -> None:
        """Denoiser_Trainer_module is ....
        Args:
            model: The model
            ckpts_file: checkpoints file
            log_dir : directory of log-file
        """

        self.model = model
        self.HW = tf.cast([-1,-1], dtype=tf.int32) # convert to tensor
        self.gpus_num=gpus_num
        self.global_batch_size = global_batch_size


        # save valid/test results
        self.return_image = True # If True, the first image of valid/test is stored



        if gpus_num>1:
            # we make `return_image` FALSE for More than one GPU
            self.return_image = False 

        super(Denoiser_Trainer_module, self).__init__(
                            ckpts_file=ckpts_file, 
                            log_dir=log_dir, 
                            strategy=strategy,
                            gpus_num=gpus_num,
                            options=options,
                            global_batch_size=self.global_batch_size, 
                            steps_per_epoch=steps_per_epoch) 

    def _compile(self,
                 optimizer: keras.optimizers.Optimizer,
                 loss: Dict[str, Callable[[tf.Tensor, tf.Tensor], tf.Tensor]],
                 loss_weights: Dict[str, float] = None,
                 metrics: Dict[str, Callable[[tf.Tensor, tf.Tensor], tf.Tensor]] = None,
                 lr=None) -> None:
        """Configure the denoiser for training.
        Args:
            optimizer: The tf.keras.optimizers optimizer for the model.
            loss: A dictionary with the losses for the model. The sum of the losses is considered.
            loss_weights: Weights of the model losses.
            metrics: evaluation metrics
        """
        self.optimizer = optimizer
        self.loss = loss
        self.loss_weights = _loss_weights(loss_weights, loss)
        self.net = 'denoiser'
        self.lr = lr

        #metrics: The metrics that should be evaluated.
        if metrics is not None: 
            self.metrics = metrics
        else:
            self.metrics = {}

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

        # create pyramid from ground-truth data. In this study, our pyramid has 
        # three scales inclduing x0 (input image), x2 (half-image) and x4 (quadratic image)
        # >>>> NOTE that `2x is meant 2 times` and `x2 is meant  1/2` <<<<
        x_gt_x2 = tf.keras.layers.AveragePooling2D(pool_size=2)(x_gt)
        x_gt_x4 = tf.keras.layers.AveragePooling2D(pool_size=4)(x_gt)


        # initilize summary 
        summary = {}
        loss_dict = {}
        [loss_dict.update({key: tf.constant(0, tf.float32)}) for key in self.loss.keys()]
        loss = tf.constant(0, tf.float32)
        with tf.GradientTape() as tape: # Tape the gradients
            # Run the model
            yhat, yhat_x2, yhat_x4 = self.model(y, training=True)

            # Compute the loss
            for loss_id, loss_fn in self.loss.items():
                
                # compute loss of each scale and then sum up
                loss_dict[loss_id]  = loss_fn(x_gt,    yhat   , self.global_batch_size) # at scale 0
                loss_dict[loss_id] += loss_fn(x_gt_x2, yhat_x2, self.global_batch_size) # at scale x2
                loss_dict[loss_id] += loss_fn(x_gt_x4, yhat_x4, self.global_batch_size) # at scale x4

            # apply weights    
            for loss_id, loss_ in loss_dict.items():    
                loss += self.loss_weights[loss_id] * loss_ 
                #tf.print(loss_id, loss_, self.loss_weights[loss_id])

                # log train results
                summary['TRAIN/' + loss_id] = loss_

        # log the total loss
        summary['TRAIN/total'] = loss

        # get the gradients
        grad = tape.gradient(loss, self.model.trainable_variables)

        # apply the gradinets
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        return summary



    #@tf.function
    def val_test_one_step(self, 
                data:tf.data.Dataset, 
                step:tf.Variable,
                prefix: str) : #-> [Dict[str, tf.Tensor], Dict[str, tf.Tensor]]
        """Run the denoiser for one step on val/test dataset.
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

    #@tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32)])
    def _denoiser_model(self, x: tf.Tensor) -> tf.Tensor:
        return self.model(x)

    def _test_summary(self, test_data: Dict[str, tf.data.Dataset]) -> Dict[str, tf.Tensor]:
        return evaluate_test(self._denoiser_model, test_data, self.HW, 'denoiser', self.metrics, 
                                self.global_batch_size, self.steps_per_epoch['TEST'])


def _loss_weights(loss_weights: Union[None, Dict[str, float]],
                  loss: Dict[str, Any]) -> Dict[str, float]:
    weights_ = {fn_name: 1. for fn_name in loss}
    if loss_weights is not None:
        weights_.update(loss_weights)
    return weights_



