'''Tensorflow implementations of callbacks for training'''

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as callbacks


class IncreaseKLDOnEpoch(callbacks.Callback):
    '''Increase VAE KLD loss linearly during training'''
    def __init__(
        self,
        factor = 1.5,
        max_val = 1.,
        **kwargs
    ):
        super().__init__()
        self.factor = factor
        self.max_val = max_val

    def on_epoch_end(self, epoch, logs=None):
        if not hasattr(self.model, 'encoder'):
          raise ValueError('Model must have a "encoder" attribute.')
        if not hasattr(self.model.encoder, 'kld_weight'):
          raise ValueError('Model encoder must have a "kld_weight" attribute.')

        kld_weight = float(K.get_value(self.model.encoder.kld_weight))
        kld_weight = min(self.factor * kld_weight, self.max_val)
        K.set_value(self.model.encoder.kld_weight, K.get_value(kld_weight))


class KLDivergenceScheduler(callbacks.Callback):
    '''Schedule VAE KLD loss during training'''
    def __init__(
        self,
        schedule,
        **kwargs
    ):
        super().__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model, 'encoder'):
          raise ValueError('Model must have a "encoder" attribute.')
        if not hasattr(self.model.encoder, 'kld_weight'):
          raise ValueError('Model encoder must have a "kld_weight" attribute.')

        kld_weight = float(K.get_value(self.model.encoder.kld_weight))
        kld_weight = self.schedule(epoch, kld_weight)
        K.set_value(self.model.encoder.kld_weight, K.get_value(kld_weight))


class CriticWeightScheduler(callbacks.Callback):
    '''Schedule critic weight during training'''
    def __init__(
        self,
        schedule,
        **kwargs
    ):
        super().__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model, 'critic_weight'):
          raise ValueError('Model must have a "critic_weight" attribute.')

        critic_weight = float(K.get_value(self.model.critic_weight))
        critic_weight = self.schedule(epoch, critic_weight)
        K.set_value(self.model.critic_weight, K.get_value(critic_weight))
