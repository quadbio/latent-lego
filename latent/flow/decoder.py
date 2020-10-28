'''Tensorflow implementations of decoder models'''

import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Dense, Activation

from .core import CoreNetwork
from .activations import clipped_softplus, clipped_exp
from .layers import ColwiseMult


class Decoder(CoreNetwork):
    '''Classical encoder model'''
    def __init__(self, x_dim, name='decoder', **kwargs):
        super().__init__(name=name, **kwargs)
        self.x_dim = x_dim
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
            self.x_dim, name = 'decoder_final',
            kernel_initializer = self.initializer,
            kernel_regularizer = l1_l2(self.l1, self.l2)
        )
        self.final_act = Activation('linear', name='reconstruction_output')


class CountDecoder(Decoder):
    '''
    Count decoder model.
    Rough reimplementation of the basic Deep Count Autoencoder by Erslan et al. 2019
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        '''Full forward pass through model'''
        h, sf = inputs
        for layer in self.core_stack:
            h = layer(h)
        mean = self.mean_layer(h)
        outputs = self.norm_layer([mean, sf])
        return outputs

    def _final(self):
        '''Final layer of the model'''
        self.mean_layer = Dense(
            self.x_dim, name='mean',
            kernel_initializer = self.initializer,
            kernel_regularizer = l1_l2(self.l1, self.l2)
        )
        self.norm_layer = ColwiseMult(name='reconstruction_output')


class PoissonDecoder(CountDecoder):
    '''
    Poisson decoder model.
    Rough reimplementation of the poisson Deep Count Autoencoder by Erslan et al. 2019
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _final(self):
        '''Final layer of the model'''
        self.mean_layer = Dense(
            self.x_dim, name='mean',
            activation = clipped_exp,
            kernel_initializer = self.initializer,
            kernel_regularizer = l1_l2(self.l1, self.l2)
        )
        self.norm_layer = ColwiseMult(name='reconstruction_output')


class NegativeBinomialDecoder(CountDecoder):
    '''
    Negative Binomial decoder model.
    Rough reimplementation of the NB Deep Count Autoencoder by Erslan et al. 2019
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        '''Full forward pass through model'''
        h, sf = inputs
        for layer in self.core_stack:
            h = layer(h)
        mean = self.mean_layer(h)
        mean = self.norm_layer([mean, sf])
        disp = self.dispersion_layer(h)
        return [mean, disp]

    def _final(self):
        '''Final layer of the model'''
        self.mean_layer = Dense(
            self.x_dim, name='mean',
            activation = clipped_exp,
            kernel_initializer = self.initializer,
            kernel_regularizer = l1_l2(self.l1, self.l2)
        )
        self.dispersion_layer = Dense(
            self.x_dim, name='dispersion',
            activation = clipped_softplus,
            kernel_initializer = self.initializer,
            kernel_regularizer = l1_l2(self.l1, self.l2)
        )
        self.norm_layer = ColwiseMult()


class ZINBDecoder(CountDecoder):
    '''
    ZINB decoder model.
    Rough reimplementation of the ZINB Deep Count Autoencoder by Erslan et al. 2019
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        '''Full forward pass through model'''
        h, sf = inputs
        for layer in self.core_stack:
            h = layer(h)
        mean = self.mean_layer(h)
        mean = self.norm_layer([mean, sf])
        disp = self.dispersion_layer(h)
        pi = self.pi_layer(h)
        return [mean, disp, pi]

    def _final(self):
        '''Final layer of the model'''
        self.mean_layer = Dense(
            self.x_dim, name='mean',
            activation = clipped_exp,
            kernel_initializer = self.initializer,
            kernel_regularizer = l1_l2(self.l1, self.l2)
        )
        self.dispersion_layer = Dense(
            self.x_dim, name='dispersion',
            activation = clipped_softplus,
            kernel_initializer = self.initializer,
            kernel_regularizer = l1_l2(self.l1, self.l2)
        )
        self.pi_layer = Dense(
            self.x_dim, name='dispersion',
            activation = 'sigmoid',
            kernel_initializer = self.initializer,
            kernel_regularizer = l1_l2(self.l1, self.l2)
        )
        self.norm_layer = ColwiseMult()
