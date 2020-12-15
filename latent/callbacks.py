"""Tensorflow implementations of callbacks for training"""

import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as callbacks
from typing import Callable


class IncreaseKLDOnEpoch(callbacks.Callback):
    """Increase Kullback-Leibler Divergence loss of VAEs during training."""
    def __init__(
        self,
        factor: float = 1.5,
        max_val: float = 1.,
        **kwargs
    ):
        """
        Arguments:
            factor: Positive float. Factor by which the KLD will be increased each epoch.
            max_val: Positive float. Maximum value of KLD.
            **kwargs: Other parameters passed to `keras.callbacks.Callback`.
        """
        super().__init__(**kwargs)
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
    """Schedule Kullback-Leibler Divergence loss of VAEs during training."""
    def __init__(
        self,
        schedule: Callable,
        **kwargs
    ):
        """
        Arguments:
            schedule: a function that takes an epoch index (integer, indexed from 0) and
                current KLD (float) as inputs and returns a new KLD as output (float).
            **kwargs: Other parameters passed to `keras.callbacks.Callback`.
        """
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
    """Schedule critic weight during training"""
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
