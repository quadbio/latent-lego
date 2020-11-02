'''Tensorflow implementations of encoder models'''

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Dense, Activation

import tensorflow_probability as tfp
tfpl = tfp.layers
tfd = tfp.distributions

from .core import DenseStack
from .activations import clipped_softplus, clipped_exp
from .layers import ColwiseMult


class Encoder(Model):
    '''Classical encoder model'''
    def __init__(
        self,
        latent_dim = 50,
        name='encoder',
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
        self.latent_dim = latent_dim
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
        self.final_layer = Dense(
            self.latent_dim, name = 'encoder_final',
            kernel_initializer = self.initializer
        )
        self.final_act = Activation('linear', name='encoder_final_activation')

    def call(self, inputs):
        '''Full forward pass through model'''
        h = self.dense_stack(inputs)
        h = self.final_layer(h)
        outputs = self.final_act(h)
        return outputs


class VariationalEncoder(Encoder):
    '''Variational encoder'''
    def __init__(self, beta=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

        # Define components
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

    def call(self, inputs):
        '''Full forward pass through model'''
        h = self.dense_stack(inputs)
        h = self.mu_sigma(h)
        outputs = self.sampling(h)
        return outputs
