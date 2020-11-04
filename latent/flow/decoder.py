'''Tensorflow implementations of decoder models'''

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1_l2
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses

from .activations import clipped_softplus, clipped_exp
from .layers import ColwiseMult, DenseStack


class Decoder(Model):
    '''Classical encoder model'''
    def __init__(
        self,
        x_dim,
        name = 'decoder',
        dropout_rate = 0.1,
        batchnorm = True,
        l1 = 0.0,
        l2 = 0.0,
        hidden_units = [128, 128],
        activation = 'leaky_relu',
        initializer = 'glorot_normal',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.x_dim = x_dim
        self.dropout_rate = dropout_rate
        self.batchnorm =  batchnorm
        self.l1 = l1
        self.l2 = l2
        self.activation = activation
        self.hidden_units =  hidden_units
        self.initializer = keras.initializers.get(initializer)

        # Define components
        self.dense_stack = DenseStack(
            name = self.name,
            dropout_rate = self.dropout_rate,
            batchnorm = self.batchnorm,
            l1 = self.l1,
            l2 = self.l2,
            activation = self.activation,
            initializer = self.initializer,
            hidden_units = self.hidden_units
        )
        self.final_layer = layers.Dense(
            self.x_dim, name = 'decoder_final',
            kernel_initializer = self.initializer
        )
        self.final_act = layers.Activation('linear', name='reconstruction_output')

    def call(self, inputs):
        '''Full forward pass through model'''
        h = self.dense_stack(inputs)
        h = self.final_layer(h)
        outputs = self.final_act(h)
        return outputs


class CountDecoder(Decoder):
    '''
    Count decoder model.
    Rough reimplementation of the poisson Deep Count Autoencoder by Erslan et al. 2019
    '''
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
        '''Full forward pass through model'''
        h, sf = inputs
        h = self.dense_stack(h)
        mean = self.mean_layer(h)
        outputs = self.norm_layer([mean, sf])
        return outputs



class NegativeBinomialDecoder(CountDecoder):
    '''
    Negative Binomial decoder model.
    Rough reimplementation of the NB Deep Count Autoencoder by Erslan et al. 2019
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean_layer = Dense(
            self.x_dim, name='mean',
            activation = clipped_exp,
            kernel_initializer = self.initializer
        )
        self.dispersion_layer = layers.Dense(
            self.x_dim, name='dispersion',
            activation = clipped_softplus,
            kernel_initializer = self.initializer
        )
        self.norm_layer = ColwiseMult()

    def call(self, inputs):
        '''Full forward pass through model'''
        h, sf = inputs
        h = self.dense_stack(h)
        mean = self.mean_layer(h)
        mean = self.norm_layer([mean, sf])
        disp = self.dispersion_layer(h)
        return [mean, disp]


class ZINBDecoder(CountDecoder):
    '''
    ZINB decoder model.
    Rough reimplementation of the ZINB Deep Count Autoencoder by Erslan et al. 2019
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean_layer = layers.Dense(
            self.x_dim, name='mean',
            activation = clipped_exp,
            kernel_initializer = self.initializer
        )
        self.dispersion_layer = layers.Dense(
            self.x_dim, name='dispersion',
            activation = clipped_softplus,
            kernel_initializer = self.initializer
        )
        self.pi_layer = layers.Dense(
            self.x_dim, name='dispersion',
            activation = 'sigmoid',
            kernel_initializer = self.initializer
        )
        self.norm_layer = ColwiseMult()

    def call(self, inputs):
        '''Full forward pass through model'''
        h, sf = inputs
        h = self.dense_stack(h)
        mean = self.mean_layer(h)
        mean = self.norm_layer([mean, sf])
        disp = self.dispersion_layer(h)
        pi = self.pi_layer(h)
        return [mean, disp, pi]
