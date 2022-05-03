# Please note that
# The functions inside `lr.py' were copied from Tensorflow source code.
# https://github.com/keras-team/keras/blob/v2.8.0/keras/callbacks.py#L2641-L2763
# So please cite and mention Tensorflow in your own studies.

from abc import ABC
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend 
import numpy as np



class ReduceLROnPlateau(ABC):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.
    Example:
    ```python
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=5, min_lr=0.001)
    ```
    Args:
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will be reduced.
                `new_lr = lr * factor`.
        patience: number of epochs with no improvement after which learning rate
            will be reduced.
        mode: one of `{'auto', 'min', 'max'}`. In `'min'` mode,
            the learning rate will be reduced when the
            quantity monitored has stopped decreasing; in `'max'` mode it will be
            reduced when the quantity monitored has stopped increasing; in `'auto'`
            mode, the direction is automatically inferred from the name of the
            monitored quantity.
        min_delta: threshold for measuring the new optimum, to only focus on
            significant changes.
        cooldown: number of epochs to wait before resuming normal operation after
                lr has been reduced.
        min_lr: lower bound on the learning rate.
    """

    def __init__(self,
            optimizer:keras.optimizers.Optimizer,
            monitor: str=None,
            factor: float=0.1,
            patience: int=10,
            mode: str='auto',
            min_delta: float=1e-4,
            cooldown: int=0,
            min_lr: int=1e-6,
            **kwargs):
        super(ReduceLROnPlateau, self).__init__()

        self.optimizer = optimizer
        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError(
                f'ReduceLROnPlateau does not support a factor >= 1.0. Got {factor}')
        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            logging.warning('`epsilon` argument is deprecated and '
                      'will be removed, use `min_delta` instead.')
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None

        # reset the LR
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            logging.warning('Learning rate reduction mode %s is unknown, '
                      'fallback to auto mode.', self.mode)
            self.mode = 'auto'
        if (self.mode == 'min' or
            (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()


    def on_epoch_end(self, step, logs=None):
        logs = logs or {}
        logs['lr'] = np.float32(backend.get_value(self.optimizer.lr))
        current = logs[self.monitor]

        if current is None:
            return logs
        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0


        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        elif not self.in_cooldown():
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = backend.get_value(self.optimizer.lr)
                if old_lr > np.float32(self.min_lr):
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    backend.set_value(self.optimizer.lr, new_lr)
                    self.cooldown_counter = self.cooldown
                    self.wait = 0
        return logs

    def in_cooldown(self):
        return self.cooldown_counter > 0