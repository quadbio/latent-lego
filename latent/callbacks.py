"""Tensorflow implementations of callbacks for training"""

import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as callbacks
from typing import Callable


class ScaleCapacityOnEpoch(callbacks.Callback):
    """Scale KLD capacity of VAEs during training."""

    def __init__(self, max_val: float = 10.0, steps: int = 100, **kwargs):
        """
        Arguments:
            max_val: Positive float. Maximum value of the capacity.
            steps: Number of steps before reaching the maximum value.
            **kwargs: Other parameters passed to `keras.callbacks.Callback`.
        """
        super().__init__(**kwargs)
        self.steps = steps
        self.max_val = max_val

    def on_epoch_end(self, epoch, logs=None):
        if not hasattr(self.model, "encoder"):
            raise ValueError('Model must have a "encoder" attribute.')
        if not hasattr(self.model.encoder, "capacity"):
            raise ValueError('Model encoder must have a "capacity" attribute.')

        capacity = float(K.get_value(self.model.encoder.capacity))
        assert capacity <= self.max_val
        delta = self.max_val - capacity
        capacity = min(capacity + delta * epoch / self.steps, self.max_val)
        K.set_value(self.model.encoder.capacity, K.get_value(capacity))


class ScaleKLDOnEpoch(callbacks.Callback):
    """Scale Kullback-Leibler Divergence loss of VAEs during training."""

    def __init__(self, max_val: float = 5.0, steps: int = 100, **kwargs):
        """
        Arguments:
            max_val: Positive float. Maximum value of KLD.
            steps: Number of steps before reaching the maximum value.
            **kwargs: Other parameters passed to `keras.callbacks.Callback`.
        """
        super().__init__(**kwargs)
        self.steps = steps
        self.max_val = max_val

    def on_epoch_end(self, epoch, logs=None):
        if not hasattr(self.model, "encoder"):
            raise ValueError('Model must have a "encoder" attribute.')
        if not hasattr(self.model.encoder, "kld_weight"):
            raise ValueError('Model encoder must have a "kld_weight" attribute.')

        kld_weight = float(K.get_value(self.model.encoder.kld_weight))
        assert kld_weight <= self.max_val
        delta = self.max_val - kld_weight
        kld_weight = min(kld_weight + delta * epoch, self.max_val)
        K.set_value(self.model.encoder.kld_weight, K.get_value(kld_weight))


class KLDivergenceScheduler(callbacks.Callback):
    """Schedule Kullback-Leibler Divergence loss of VAEs during training."""

    def __init__(self, schedule: Callable, **kwargs):
        """
        Arguments:
            schedule: a function that takes an epoch index (integer, indexed from 0) and
                current KLD (float) as inputs and returns a new KLD as output (float).
            **kwargs: Other parameters passed to `keras.callbacks.Callback`.
        """
        super().__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model, "encoder"):
            raise ValueError('Model must have a "encoder" attribute.')
        if not hasattr(self.model.encoder, "kld_weight"):
            raise ValueError('Model encoder must have a "kld_weight" attribute.')

        kld_weight = float(K.get_value(self.model.encoder.kld_weight))
        kld_weight = self.schedule(epoch, kld_weight)
        K.set_value(self.model.encoder.kld_weight, K.get_value(kld_weight))


class CriticWeightScheduler(callbacks.Callback):
    """Schedule critic weight during training"""

    def __init__(self, schedule, **kwargs):
        super().__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model, "critic_weight"):
            raise ValueError('Model must have a "critic_weight" attribute.')

        critic_weight = float(K.get_value(self.model.critic_weight))
        critic_weight = self.schedule(epoch, critic_weight)
        K.set_value(self.model.critic_weight, K.get_value(critic_weight))
