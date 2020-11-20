"""Tensorflow implementations of decoder models"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1_l2
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses

from .activations import clipped_softplus, clipped_exp
from .layers import ColwiseMult, DenseStack, SharedDispersion, Constant


class Decoder(keras.Model):
    """Deocder base model"""
    def __init__(
        self,
        x_dim,
        name = 'decoder',
        dropout_rate = 0.1,
        batchnorm = True,
        l1 = 0.,
        l2 = 0.,
        hidden_units = [128, 128],
        activation = 'leaky_relu',
        initializer = 'glorot_normal',
        **kwargs
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
        h = self.dense_stack(inputs) if self.hidden_units else inputs
        h = self.final_layer(h)
        outputs = self.final_act(h)
        return outputs

    def hidden(self, inputs):
        """Pass through hidden layers"""
        return self.hidden_layers(inputs) if self.hidden_units else inputs

    def output(self, inputs):
        """Resonstructs output"""
        return self.final_layer(inputs)


class CountDecoder(Decoder):
    """
    Count decoder model.
    Rough reimplementation of the poisson Deep Count Autoencoder by Erslan et al. 2019
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Define new components
        self.mean_layer = layers.Dense(
            self.x_dim, name='mean',
            activation = clipped_exp,
            kernel_initializer = self.initializer
        )
        self.norm_layer = ColwiseMult(name='reconstruction_output')

    def call(self, inputs):
        """Full forward pass through model"""
        h, sf = inputs
        h = self.dense_stack(h)
        mean = self.mean_layer(h)
        outputs = self.norm_layer([mean, sf])
        return outputs


class NegativeBinomialDecoder(Decoder):
    """
    Negative Binomial decoder model.
    Rough reimplementation of the NB Deep Count Autoencoder by Erslan et al. 2019
    """
    def __init__(self, dispersion='gene', **kwargs):
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
        elif isinstance(dispersion, (float, int)):
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
        h, sf = inputs
        h = self.dense_stack(h)
        mean = self.mean_layer(h)
        mean = self.norm_layer([mean, sf])
        disp = self.dispersion_layer(h)
        return [mean, disp]


class ZINBDecoder(NegativeBinomialDecoder):
    """
    ZINB decoder model.
    Rough reimplementation of the ZINB Deep Count Autoencoder by Erslan et al. 2019
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Define new components
        self.pi_layer = layers.Dense(
            self.x_dim, name='dispersion',
            activation = 'sigmoid',
            kernel_initializer = self.initializer
        )

    def call(self, inputs):
        """Full forward pass through model"""
        h, sf = inputs
        h = self.dense_stack(h)
        mean = self.mean_layer(h)
        mean = self.norm_layer([mean, sf])
        disp = self.dispersion_layer(h)
        pi = self.pi_layer(h)
        return [mean, disp, pi]
