'''Tensorflow Autoencoder Model'''

import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Activation, Lambda, Dropout, LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Dense


class Encoder(Model):
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

        self.input_layer = Input(shape=(self.x_dim, ), name='data')
        self.model = self._model()

    def _model(self):
        '''Constructs the full model network'''
        h = self.input_layer
        for idx, dim in enumerate(self.architecture):
            layer_name = f'encoder_{idx}'
            h = Dense(
                dim, name = layer_name,
                kernel_initializer = self.initializer,
                kernel_regularizer = l1_l2(self.l1, self.l2)
            )(h)

            if self.batchnorm:
                h = BatchNormalization(center=True, scale=False)(h)

            h = LeakyReLU()(h)

            if self.dropout_rate > 0.0:
                h = Dropout(self.dropout_rate)(h)

        latent = Dense(
            self.latent_dim, name = 'encoder_final',
            kernel_initializer = self.initializer,
            kernel_regularizer = l1_l2(self.l1, self.l2)
        )(h)

        model = Model(inputs=self.input_layer, outputs=latent, name='encoder')
        return model

    def call(self, inputs):
        return self.model(inputs)


class Decoder(Model):
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

        self.input_layer = Input(shape=(self.latent_dim, ), name='latent')
        self.model = self._model()

    def _model(self):
        '''Constructs the full model network'''
        h = self.input_layer
        for idx, dim in enumerate(self.architecture):
            layer_name = f'decoder_{idx}'
            h = Dense(
                dim, name = layer_name,
                kernel_initializer = self.initializer,
                kernel_regularizer = l1_l2(self.l1, self.l2)
            )(h)

            if self.batchnorm:
                h = BatchNormalization(center=True, scale=False)(h)

            h = LeakyReLU()(h)

            if self.dropout_rate > 0.0:
                h = Dropout(self.dropout_rate)(h)

        h = Dense(
            self.x_dim, name = 'decoder_final',
            kernel_initializer = self.initializer,
            kernel_regularizer = l1_l2(self.l1, self.l2)
        )(h)
        reconstructed = Activation('linear', name='reconstruction_output')(h)
        model = Model(inputs=self.input_layer, outputs=reconstructed, name='decoder')
        return model

    def call(self, inputs):
        return self.model(inputs)
