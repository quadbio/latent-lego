'''Tensorflow implementations of encoder models'''

import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Activation, Lambda, Dropout, LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Dense

from .activations import clipped_softplus, clipped_exp
from .layers import ColwiseMult


class Encoder(Model):
    '''Classical encoder model'''

    def __init__(
        self,
        x_dim,
        latent_dim = 50,
        dropout_rate = 0.1,
        batchnorm = True,
        l1 = 0.0,
        l2 = 0.0,
        architecture = [128, 128],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.x_dim = x_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.batchnorm =  batchnorm
        self.l1 = l1
        self.l2 = l2
        self.architecture =  architecture
        self.initializer = keras.initializers.glorot_normal()

        self._core()
        self._final()

    def call(self, inputs):
        '''Full forward pass through model'''
        h = inputs
        for layer in self.core_stack:
            h = layer(h)
        h = self.final_layer(h)
        outputs = self.final_act(h)
        return outputs

    def _core(self):
        '''Core layers of the model'''
        self.core_stack = []
        for idx, dim in enumerate(self.architecture):
            layer_name = f'encoder_{idx}'
            dense = Dense(
                dim, name = layer_name,
                kernel_initializer = self.initializer,
                kernel_regularizer = l1_l2(self.l1, self.l2)
            )
            self.core_stack.append(dense)

            if self.batchnorm:
                bn = BatchNormalization(center=True, scale=False)
                self.core_stack.append(bn)

            lrelu = LeakyReLU()
            self.core_stack.append(lrelu)

            if self.dropout_rate > 0.0:
                dout = Dropout(self.dropout_rate)
                self.core_stack.append(dout)

    def _final(self):
        '''Final layer of the model'''
        self.final_layer = Dense(
            self.latent_dim, name = 'encoder_final',
            kernel_initializer = self.initializer,
            kernel_regularizer = l1_l2(self.l1, self.l2)
        )
        self.final_act = Activation('linear', name='encoder_final_activation')
