'''Tensorflow implementations of decoder models'''

import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout

from .activations import ACTIVATIONS


class CoreNetwork(Model):
    '''Core network of encoders and decoders'''
    def __init__(
        self,
        name = 'net',
        dropout_rate = 0.1,
        batchnorm = True,
        l1 = 0.0,
        l2 = 0.0,
        architecture = [128, 128],
        activation = 'leaky_relu',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.dropout_rate = dropout_rate
        self.batchnorm =  batchnorm
        self.l1 = l1
        self.l2 = l2
        self.architecture =  architecture
        self.initializer = keras.initializers.glorot_normal()

        if isinstance(activation, str):
            self.activation = ACTIVATIONS.get(activation, LeakyReLU())
        else:
            self.activation = activation

        self._core()

    def call(self, inputs):
        '''Full forward pass through model'''
        h = inputs
        for layer in self.core_stack:
            h = layer(h)
        outputs = h
        return outputs

    def _core(self):
        '''Core layer stack'''
        self.core_stack = []
        for idx, dim in enumerate(self.architecture):
            layer_name = f'd{self.name}_{idx}'
            layer = Dense(
                dim, name = layer_name,
                kernel_initializer = self.initializer,
                kernel_regularizer = l1_l2(self.l1, self.l2)
            )
            self.core_stack.append(layer)

            if self.batchnorm:
                layer = BatchNormalization(center=True, scale=True)
                self.core_stack.append(layer)

            self.core_stack.append(self.activation)

            if self.dropout_rate > 0.0:
                layer = Dropout(self.dropout_rate)
                self.core_stack.append(layer)
