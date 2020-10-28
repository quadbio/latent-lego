'''Tensorflow implementations of encoder models'''

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Dense, Activation

import tensorflow_probability as tfp
tfpl = tfp.layers
tfd = tfp.distributions

from .core import CoreNetwork
from .activations import clipped_softplus, clipped_exp
from .layers import ColwiseMult


class Encoder(CoreNetwork):
    '''Classical encoder model'''
    def __init__(self, latent_dim=50, name='encoder', **kwargs):
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self._final()

    def call(self, inputs):
        '''Full forward pass through model'''
        h = inputs
        for layer in self.core_stack:
            h = layer(h)
        h = self.final_layer(h)
        outputs = self.final_act(h)
        return outputs

    def _final(self):
        '''Final layer of the model'''
        self.final_layer = Dense(
            self.latent_dim, name = 'encoder_final',
            kernel_initializer = self.initializer,
            kernel_regularizer = l1_l2(self.l1, self.l2)
        )
        self.final_act = Activation('linear', name='encoder_final_activation')


class VariationalEncoder(Encoder):
    '''Variational encoder'''
    def __init__(self, beta=0.1, **kwargs):
        self.beta = beta
        super().__init__(**kwargs)

    def call(self, inputs):
        '''Full forward pass through model'''
        h = inputs
        for layer in self.core_stack:
            h = layer(h)
        h = self.mu_sigma(h)
        outputs = self.sampling(h)
        return outputs

    def _final(self):
        '''Final layer of the model'''
        self.mu_sigma = Dense(
            tfpl.MultivariateNormalTriL.params_size(self.latent_dim),
            name = 'encoder_mu_sigma',
            kernel_initializer = self.initializer,
            kernel_regularizer = l1_l2(self.l1, self.l2)
        )
        # Make priors more flexible in the future
        # Independent() reinterprets each latent_dim as an independent distribution
        self.prior = tfd.Independent(
            tfd.Normal(loc=tf.zeros(self.latent_dim), scale=1),
            reinterpreted_batch_ndims = 1
        )
        self.sampling = tfpl.MultivariateNormalTriL(
            self.latent_dim,
            activity_regularizer = tfpl.KLDivergenceRegularizer(
                self.prior,
                weight = self.beta
            )
        )
