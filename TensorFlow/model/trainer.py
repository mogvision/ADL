from typing import Any, Callable, Dict, Tuple, Union
from tensorflow import keras
import tensorflow as tf
from abc import ABC
import numpy as np
import itertools
import tempfile

from datetime import datetime
tf.config.experimental.enable_tensor_float_32_execution(True)
#tf.config.experimental.set_synchronous_execution(False)



class Multiple_Steps_Trainer(ABC):
    def __init__(self, 
        ckpts_file: str, 
                log_dir: str, 
                strategy, 
                gpus_num:int, 
                options,
                global_batch_size:int,
                steps_per_epoch:Dict
                ):
        """A trainer which can be trained with the Keras API.

        Args:
            ckpts_file: Path to the checkpoints file
            log_dir: Directory for the tensorboard logging
        """
        super(Multiple_Steps_Trainer, self).__init__()

        self.ckpts_file = ckpts_file
        self.log_dir = log_dir
        self.strategy = strategy
        self.options = options
        self.gpus_num = gpus_num
        self.global_batch_size = global_batch_size 
        self.steps_per_epoch = steps_per_epoch 

    def compile(self, *args, **kwargs):
        """Configures the trainer for training."""

        self._compile(*args, **kwargs)
        self.lr.on_train_begin()
        self.compiled = True

    def fit(self,
            train_data: tf.data.Dataset,
            validation_data: tf.data.Dataset=None,
            test_data: Dict[str, tf.data.Dataset] = None,
            steps: int=0,
            initial_step: int=0,
            validation_per_step: int=0,
            test_per_step: int=0,
            checkpoints_per_step: int=0):

        # check whether compiled was excuted ...
        if not self.compiled:
            raise ValueError("Please call compile() before training!")

        # Creates summary file writer for the given log directory.
        self.summary_writer_numerical  = self._create_file_writer(f"{self.log_dir}/numerical", 'write_log' ) 
        self.summary_writer_img  = self._create_file_writer(f"{self.log_dir}/images",'write_image') 

        # convert the counting params into Tensor
        steps_ = tf.convert_to_tensor(steps,dtype=tf.int64)
        iterator = iter(train_data)

        with self.strategy.scope():
            step = tf.Variable(1, dtype=tf.int64)

            while step < steps_: 
                try:
                    data_inp = next(iterator)

                    #assert tf.distribute.get_replica_context() is  None 
                    ############################################
                    #                1/4: Training
                    ############################################
                    summary = self.strategy.run(self.train_one_step, args=(data_inp, step,))

                    # After apply `self.strategy.run` on data, we need to convert Replica to Tensor. 
                    # In the case of gpu >1: 
                    #      v.values = [x_from_dev_0, x_from_dev_1, ...]  axis=0 -> batch dim
                    for key, value in summary.items():
                        values_agg = value if self.gpus_num <= 1 else tf.concat(value.values, axis=0)  
                        summary[key] = tf.reduce_mean(values_agg)
                
                    # store results
                    self._log(summary, step)

                    ############################################
                    #                2/4: Validation
                    ############################################
                    if step%validation_per_step == 0:

                        #replica_context = tf.distribute.get_replica_context()  # for strategy
                        #assert replica_context is not None

                        # print results
                        tf.print()
                        step_i = step if self.gpus_num <= 1 else tf.concat(step.values, axis=0)[0]
                        tf.print(f'{datetime.now()} \t step: <<<<< ', step_i, '/', int(tf.divide(steps_, self.gpus_num)), ' >>>>>')
                        self._print_summary('TRAIN:\t', summary)


                        if validation_data is not None:
                            val_summary = self._validation_test(validation_data, step, 'VAL')
                            val_summary = self.lr.on_epoch_end(step, val_summary)
                            self._log(val_summary, step) 
                            self._print_summary('VALIDATION:\t', val_summary)


                    ############################################
                    #                 3/4: Testing
                    ############################################
                    if step%test_per_step == 0 and test_data is not None: 
                        summary = self._validation_test(test_data, step, 'TEST')
                        self._log(summary, step) 
                        self._print_summary('TEST:\t', summary)

                    ############################################
                    #              4/4: Storing model
                    ############################################
                    if step % checkpoints_per_step == 0 :

                        stepid = step // checkpoints_per_step
                        self._sava_model(stepid)
                        tf.print(f"[i] checkpoint is saved with id <{stepid}>.") 

                except StopIteration:
                    break

                step.assign_add(1)


    #*****************************************************************
    #                       auxiliary funcs
    #*****************************************************************
    def _validation_test(self, 
                    data:tf.data.Dataset, 
                    Step:tf.Variable,
                    mode:str) -> Dict[str, tf.Tensor]: 
        """
            ......
        """
        summary = {}

        # Initialize with 0
        for param in self.metrics:
            summary[f"{mode}/{param}"] = tf.constant(0., dtype=tf.float32)


        # Loop over the data and sum up metric
        if mode == 'TEST':
            ds_size = self.steps_per_epoch['TEST']
        elif mode == 'VAL':
            ds_size = self.steps_per_epoch['VAL']
        else:
            raise NotImplementedError("Dataset is not defined in this code!")


        ds_size = tf.convert_to_tensor(ds_size)
        iter_data = iter(data)
        counter, step = tf.Variable(0., dtype=tf.float32), tf.Variable(0, dtype=ds_size.dtype)

        while  step < ds_size:
            try:
                data_inp = next(iter_data)
                summary_tmp, img_summary = self.strategy.run(self.val_test_one_step, args=(data_inp, step, mode))

                # save numerical results
                if summary_tmp is not None:
                    for key, value in summary_tmp.items():

                        # collect the results from all devices
                        values_agg = value if self.gpus_num <= 1 else tf.concat(value.values, axis=0) 
                        summary[key] += tf.reduce_sum(values_agg)
                    counter.assign_add(1)

                # save tesultant images
                if img_summary is not None:
                    self._log_images(img_summary, Step)

            except StopIteration:
                break

            step.assign_add(1)

        # Compute the average of each metric
        for param in summary.keys():
            summary[param] = tf.divide(summary[param], counter) 

        return summary



    def _create_file_writer(self, log_dir, name): 
        return tf.summary.create_file_writer(log_dir, name=name)

    def _log(self, summary, step):
        if summary is not None:
            with self.summary_writer_numerical.as_default():
                for key, value in summary.items():
                    tf.summary.scalar(key, value, step=step)

    def _log_images(self, img_summary, step):
        with self.summary_writer_img.as_default():
            for img_id, imgs in img_summary.items():
                imgs_agg = imgs if self.gpus_num <= 1 else tf.concat(imgs.values, axis=0) 
                tf.summary.image(img_id, tf.expand_dims(imgs_agg[0,...], axis=0), step=step)
    
    def _sava_model(self, stepid):
        with self.strategy.scope():
            self.model.save(f'{self.ckpts_file}/checkpoint-{stepid}', 
                                    save_format='tf', include_optimizer=True,
                                    options=self.options.save_options)


    def _print_summary(self, prefix, summary):
        precision = '0.5f'
        tf.print(f'{prefix.ljust(11)}-> ', end = '') 
        keys_num = len(summary.keys())

        lr_ = None
        counter = 1
        for key, val in summary.items():
            key = key.split('/')[-1]
            value= format(tf.reduce_mean(val), precision)
            if key in ['lr', 'LR']:
                lr_ = format(tf.reduce_mean(val), '0.9f')
                continue
            if counter < keys_num:
                tf.print(f"{key.ljust(4)}: {value.ljust(8, '0')}", end = ' | ')
            else:
                tf.print(f"{key.ljust(4)}: {value.ljust(8, '0')}", end = ' ')
            counter += 1

        if lr_ is not None:
            tf.print(f"\nLearning rate: {lr_}", end = ' ')

        tf.print()
