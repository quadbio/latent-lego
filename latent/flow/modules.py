'''Tensorflow Autoencoder Model'''

import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Activation, Lambda, Dropout, LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Dense

from .activations import clipped_softplus, clipped_exp
from .layers import ColwiseMult, Slice

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

        self.input_layer = Input(shape=(self.x_dim, ), name='data')
        inputs, outputs = self._build_model()
        self.model = Model(inputs=inputs, outputs=outputs, name='encoder')

    def _build_model(self):
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

        inputs, outputs = self._build_final(h)
        return inputs, outputs

    def _build_final(self, inputs):
        '''Final layer of the model'''
        h = Dense(
            self.latent_dim, name = 'encoder_build_final',
            kernel_initializer = self.initializer,
            kernel_regularizer = l1_l2(self.l1, self.l2)
        )(inputs)
        outputs = Activation('linear', name='latent')(h)
        return self.input_layer, outputs

    def call(self, inputs):
        return self.model(inputs)


class Decoder(Model):
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

        self.input_layer = Input(shape=(self.latent_dim, ), name='latent')
        inputs, outputs = self._build_model()
        self.model = Model(inputs=inputs, outputs=outputs, name='encoder')

    def _build_model(self):
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

        inputs, outputs = self._build_final(h)
        return inputs, outputs

    def _build_final(self, inputs):
        '''Final layer of the model'''
        h = Dense(
            self.x_dim, name = 'decoder_build_final',
            kernel_initializer = self.initializer,
            kernel_regularizer = l1_l2(self.l1, self.l2)
        )(inputs)
        outputs = Activation('linear', name='reconstruction_output')(h)
        return self.input_layer, outputs

    def call(self, inputs):
        return self.model(inputs)


class CountDecoder(Decoder):
    '''
    Count decoder model.
    Rough reimplementation of the basic Deep Count Autoencoder by Erslan et al. 2019
    '''

    def __init__(self, **kwargs):
        self.sf_layer = Input(shape=(1, ), name='size_factors')
        super().__init__(**kwargs)

    def _build_final(self, inputs):
        '''Final layer of the model'''
        mean = Dense(
            self.x_dim, name='mean',
            kernel_initializer = self.initializer,
            kernel_regularizer = l1_l2(self.l1, self.l2)
        )(inputs)
        outputs = ColwiseMult()([mean, self.sf_layer])

        return [self.input_layer, self.sf_layer], outputs


class PoissonDecoder(CountDecoder):
    '''
    Poisson decoder model.
    Rough reimplementation of the poisson Deep Count Autoencoder by Erslan et al. 2019
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_final(self, inputs):
        '''Final layer of the model'''
        mean = Dense(
            self.x_dim, name='mean',
            kernel_initializer = self.initializer,
            kernel_regularizer = l1_l2(self.l1, self.l2)
        )(inputs)
        mean = Activation(clipped_exp, name='clipped_exp')(mean)
        outputs = ColwiseMult()([mean, self.sf_layer])

        return [self.input_layer, self.sf_layer], outputs


class NegativeBinomialDecoder(CountDecoder):
    '''
    Poisson decoder model.
    Rough reimplementation of the poisson Deep Count Autoencoder by Erslan et al. 2019
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_final(self, inputs):
        '''Final layer of the model'''
        mean = Dense(
            self.x_dim, name='mean',
            kernel_initializer = self.initializer,
            kernel_regularizer = l1_l2(self.l1, self.l2)
        )(inputs)
        mean = Activation(clipped_exp, name='clipped_exp')(mean)

        disp = Dense(
            self.x_dim, name='dispersion',
            kernel_initializer = self.initializer,
            kernel_regularizer = l1_l2(self.l1, self.l2)
        )(inputs)
        disp = Activation(clipped_softplus, name='clipped_softplus')(disp)

        # Define dispersion model
        self.disp_model = Model(
            inputs = [self.input_layer, self.sf_layer],
            outputs = disp,
            name = 'dispersion_decoder'
        )

        # This is necessary to include disp as an intermediate layer
        # without requiring it to be an output of the model
        outputs = Slice(0)([mean, disp])
        outputs = ColwiseMult()([mean, self.sf_layer])

        return [self.input_layer, self.sf_layer], outputs
