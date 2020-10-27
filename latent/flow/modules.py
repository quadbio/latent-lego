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
        build_model = False,
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

        if build_model:
            self._build()

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

    def _build(self):
        self.build(input_shape=(self.x_dim, ))


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
        build_model = False,
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

        if build_model:
            self._build()

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
            layer_name = f'decoder_{idx}'
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
            self.x_dim, name = 'decoder_final',
            kernel_initializer = self.initializer,
            kernel_regularizer = l1_l2(self.l1, self.l2)
        )
        self.final_act = Activation('linear', name='reconstruction_output')

    def _build(self):
        self.build(input_shape=(self.latent_dim, ))


class CountDecoder(Decoder):
    '''
    Count decoder model.
    Rough reimplementation of the basic Deep Count Autoencoder by Erslan et al. 2019
    '''

    def __init__(self, **kwargs):
        self.sf_layer = Input(shape=(1, ), name='size_factors')
        super().__init__(**kwargs)

    def call(self, inputs):
        '''Full forward pass through model'''
        h = inputs[0]
        sf = inputs[1]
        for layer in self.core_stack:
            h = layer(h)
        h = self.mean_layer(h)
        outputs = self.norm_layer([h, sf])
        return outputs

    def _final(self, inputs):
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

    def _final(self, inputs):
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

    def _final(self, inputs):
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
