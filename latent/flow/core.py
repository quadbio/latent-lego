'''Tensorflow implementations of decoder models'''

import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout

from .activations import ACTIVATIONS


class DenseBlock(Layer):
    '''Basic dense layer block'''
    def __init__(
        self,
        name = 'dense_block',
        dropout_rate = 0.1,
        batchnorm = True,
        l1 = 0.0,
        l2 = 0.0,
        activation = 'leaky_relu',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.name = name
        self.dropout_rate = dropout_rate
        self.batchnorm =  batchnorm
        self.l1 = l1
        self.l2 = l2
        self.initializer = keras.initializers.glorot_normal()

        # Define block components
        self.dense = Dense(
            dim, name = self.name,
            kernel_initializer = self.initializer,
            kernel_regularizer = l1_l2(self.l1, self.l2)
        )
        self.bn = BatchNormalization(center=True, scale=True)
        if isinstance(activation, str):
            self.activation = ACTIVATIONS.get(activation, LeakyReLU())
        else:
            self.activation = activation
        self.dropout = Dropout(self.dropout_rate)


    def call(self, inputs):
        '''Full forward pass through model'''
        h = self.dense(inputs)
        h = self.bn(h)
        h = self.activation(h)
        outputs = self.dropout(h)
        return outputs


class DenseStack(Layer):
    '''Core layer stack of encoders and decoders'''
    def __init__(
        self,
        name = 'dense_stack',
        dropout_rate = 0.1,
        batchnorm = True,
        l1 = 0.0,
        l2 = 0.0,
        architecture = [128, 128],
        activation = 'leaky_relu',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.architecture =  architecture

        self.layer_stack = []
        for idx, dim in enumerate(self.architecture):
            layer_name = f'd{self.name}_{idx}'
            layer = DenseBlock(
                name = layer_name,
                dropout_rate = dropout_rate,
                batchnorm = batchnorm,
                l1 = l1,
                l2 = l2
            )
            self.layer_stack.append(layer)

    def call(self, inputs):
        '''Full forward pass through model'''
        h = inputs
        for layer in self.layer_stack:
            h = layer(h)
        outputs = h
        return outputs
