"""Tensorflow implementations of decoder models"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1_l2
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses

from typing import Iterable, Literal, Union, Callable

from latent.activations import clipped_softplus, clipped_exp
from latent.layers import ColwiseMult, DenseStack, SharedDispersion, Constant
from latent.layers import DenseBlock
from latent.losses import NegativeBinomial, ZINB


class Decoder(keras.Model):
    """Deocder base model"""
    def __init__(
        self,
        x_dim: int,
        name: str = 'decoder',
        hidden_units = [128, 128],
        reconstruction_loss: Callable = None,
        loss_name = 'rec_loss',
        initializer: Union[str, Callable] = 'glorot_normal',
        **kwargs
    ):
        super().__init__(name=name)
        self.x_dim = x_dim
        self.hidden_units =  hidden_units
        self.reconstruction_loss = reconstruction_loss
        self.loss_name = loss_name
        self.initializer = keras.initializers.get(initializer)
        # Set use_sf to False because this base model only expects one input
        self.use_sf = False

        # Define components
        if self.hidden_units:
            self.hidden_layers = DenseStack(
                name = f'{self.name}_hidden',
                hidden_units = self.hidden_units,
                initializer = self.initializer,
                **kwargs
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
        outputs = self.final_layer(h)
        self.add_reconstruction_loss(x, outputs)
        return outputs

    def hidden(self, latent):
        """Pass through hidden layers"""
        return self.hidden_layers(latent) if self.hidden_units else latent

    def add_reconstruction_loss(self, x, output):
        """Adds reconstruction loss to final model loss"""
        if self.reconstruction_loss:
            rec_loss = self.reconstruction_loss(x, output)
            self.add_loss(rec_loss)
            self.add_metric(rec_loss, name=self.loss_name)


class PoissonDecoder(Decoder):
    """Decoder with poisson reconstruction loss."""
    def __init__(
        self,
        name = 'poisson_decoder',
        reconstruction_loss: Callable = losses.Poisson(),
        loss_name: str = 'poisson_loss',
        **kwargs
    ):
        super().__init__(
            name = name,
            reconstruction_loss = reconstruction_loss,
            loss_name = loss_name,
            **kwargs
        )
        # Here use_sf becomes True
        self.use_sf = True

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
        h = self.hidden(latent)
        mean = self.mean_layer(h)
        outputs = self.norm_layer([mean, sf])
        self.add_reconstruction_loss(x, outputs)
        return outputs


class NegativeBinomialDecoder(PoissonDecoder):
    """Decoder with negative binomial reconstruction loss."""
    def __init__(
        self,
        name: str = 'nb_decoder',
        dispersion: Union[Literal['gene', 'cell-gene', 'constant'], float] = 'gene',
        loss_name: str = 'nb_loss',
        **kwargs
    ):
        super().__init__(
            name = name,
            loss_name = loss_name,
            reconstruction_loss = None,
            **kwargs
        )
        self.dispersion = dispersion
        self.use_sf = True

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
                kernel_initializer = self.initializer
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


class ZINBDecoder(NegativeBinomialDecoder):
    """Decoder with ZINB reconstruction loss"""
    def __init__(
        self,
        name: str = 'zinb_decoder',
        loss_name: str = 'zinb_loss',
        **kwargs
    ):
        super().__init__(
            name = name,
            loss_name = loss_name,
            **kwargs
        )
        self.use_sf = True

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
        h = self.hidden(latent)
        mean = self.mean_layer(h)
        outputs = self.norm_layer([mean, sf])
        disp = self.dispersion_layer(h)
        pi = self.pi_layer(h)
        self.reconstruction_loss = ZINB(theta=disp, pi=pi)
        self.add_reconstruction_loss(x, outputs)
        return outputs
