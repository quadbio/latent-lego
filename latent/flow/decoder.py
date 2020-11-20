"""Tensorflow implementations of decoder models"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1_l2
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses

from fastcore import delegates
from typing import Iterable, Literal, Union, Callable

from .activations import clipped_softplus, clipped_exp
from .layers import ColwiseMult, DenseStack, SharedDispersion, Constant
from .losses import NegativeBinomial, ZINB


class Decoder(keras.Model):
    """Deocder base model"""
    def __init__(
        self,
        x_dim: int,
        name: str = 'decoder',
        dropout_rate: float = 0.1,
        batchnorm: bool = True,
        l1: float = 0.,
        l2: float = 0.,
        hidden_units = [128, 128],
        activation: Union[str, Callable] = 'leaky_relu',
        initializer: Union[str, Callable] = 'glorot_normal',
        reconstruction_loss: Callable = None,
        loss_name = 'rec_loss'
    ):
        super().__init__(name=name)
        self.x_dim = x_dim
        self.dropout_rate = dropout_rate
        self.batchnorm =  batchnorm
        self.l1 = l1
        self.l2 = l2
        self.activation = activation
        self.hidden_units =  hidden_units
        self.initializer = keras.initializers.get(initializer)
        self.reconstruction_loss = reconstruction_loss
        self.loss_name = loss_name

        # Define components
        if self.hidden_units:
            self.hidden_layers = DenseStack(
                name = f'{self.name}_hidden',
                dropout_rate = self.dropout_rate,
                batchnorm = self.batchnorm,
                l1 = self.l1,
                l2 = self.l2,
                activation = self.activation,
                initializer = self.initializer,
                hidden_units = self.hidden_units
            )
        self.final_layer = layers.Dense(
            self.x_dim,
            name = f'{self.name}_output',
            kernel_initializer = self.initializer,
            activation = 'linear'
        )

    def call(self, inputs):
        """Full forward pass through model"""
        x, latent = inputs
        h = self.hidden(latent)
        output = self.final_layer(h)
        self.add_reconstruction_loss(x, output)
        return outputs

    def hidden(self, latent):
        """Pass through hidden layers"""
        return self.hidden_layers(latent) if self.hidden_units else latent

    def add_reconstruction_loss(self, x, output):
        """Adds reconstruction loss to final model loss"""
        if self.reconstruction_loss:
            rec_loss = self.reconstruction_loss(latent, output)
            self.add_loss(rec_loss)
            self.add_metric(rec_loss, name=self.loss_name)


@delegates()
class PoissonDecoder(Decoder):
    """Decoder with poisson reconstruction loss."""
    def __init__(
        self,
        name = 'poisson_decoder',
        reconstruction_loss: Callable = losses.Poisson(),
        loss_name: str = 'poisson_loss',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        # Define new components
        self.mean_layer = layers.Dense(
            self.x_dim,
            name = 'mean',
            activation = clipped_exp,
            kernel_initializer = self.initializer
        )
        self.norm_layer = ColwiseMult(name='output')

    def call(self, inputs):
        """Full forward pass through model"""
        x, latent, sf = inputs
        h = self.hidden(h)
        mean = self.mean_layer(h)
        outputs = self.norm_layer([mean, sf])
        self.add_reconstruction_loss(x, outputs)
        return outputs


@delegates()
class NegativeBinomialDecoder(PoissonDecoder):
    """Decoder with negative binomial reconstruction loss."""
    def __init__(
        self,
        name: str = 'nb_decoder',
        dispersion: Union[Literal['gene', 'cell-gene', 'constant'], float] = 'gene',
        loss_name: str = 'nb_loss',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dispersion = dispersion

        # Define new components
        self.mean_layer = layers.Dense(
            self.x_dim, name='mean',
            activation = clipped_exp,
            kernel_initializer = self.initializer
        )
        if dispersion == 'cell-gene':
            self.dispersion_layer = layers.Dense(
                self.x_dim,
                name = 'dispersion',
                activation = clipped_exp,
                initializer = self.initializer
            )
        elif dispersion == 'gene':
            self.dispersion_layer = SharedDispersion(
                self.x_dim,
                name = 'shared_dispersion',
                activation = clipped_exp,
                initializer = self.initializer
            )
        elif dispersion == 'constant':
            self.dispersion_layer = Constant(
                self.x_dim,
                trainable = True,
                name = 'constant_dispersion',
                activation = clipped_exp
            )
        elif isinstance(dispersion, float):
            self.dispersion_layer = Constant(
                self.x_dim,
                constant = self.dispersion,
                trainable = False,
                name = 'constant_dispersion',
                activation = clipped_exp
            )
        self.norm_layer = ColwiseMult()

    def call(self, inputs):
        """Full forward pass through model"""
        x, latent, sf = inputs
        h = self.hidden(latent)
        mean = self.mean_layer(h)
        outputs = self.norm_layer([mean, sf])
        disp = self.dispersion_layer(h)
        self.reconstruction_loss = NegativeBinomial(theta=disp)
        self.add_reconstruction_loss(x, outputs)
        return outputs


@delegates()
class ZINBDecoder(NegativeBinomialDecoder):
    """Decoder with ZINB reconstruction loss"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Define new components
        self.pi_layer = layers.Dense(
            self.x_dim,
            name = 'dropout_rate',
            activation = 'sigmoid',
            kernel_initializer = self.initializer
        )

    def call(self, inputs):
        """Full forward pass through model"""
        x, latent, sf = inputs
        h = self.dense_stack(h)
        mean = self.mean_layer(h)
        outputs = self.norm_layer([mean, sf])
        disp = self.dispersion_layer(h)
        pi = self.pi_layer(h)
        self.reconstruction_loss = ZINB(theta=disp, pi=pi)
        self.add_reconstruction_loss(x, outputs)
        return outputs
