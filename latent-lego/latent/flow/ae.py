'''Tensorflow Autoencoder Model'''

import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model

from .modules import Encoder, Decoder

class AE(Model):
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
        self.latent_dim = latent_dim
        self.x_dim = int(x_dim)
        self.dropout_rate = dropout_rate
        self.batchnorm =  batchnorm
        self.l1 = l1
        self.l2 = l2
        self.architecture =  architecture

        self.initializer = keras.initializers.glorot_normal()
        self.encoder = Encoder(
            x_dim = self.x_dim,
            latent_dim = self.latent_dim,
            dropout_rate = self.dropout_rate,
            batchnorm = self.batchnorm,
            l1 = self.l1,
            l2 = self.l2,
            architecture = self.architecture
        )
        self.decoder = Decoder(
            x_dim = self.x_dim,
            latent_dim = self.latent_dim,
            dropout_rate = self.dropout_rate,
            batchnorm = self.batchnorm,
            l1 = self.l1,
            l2 = self.l2,
            architecture = self.architecture[::-1]
        )
        self.input_layer = Input(shape=(self.x_dim, ), name='data')
        self.model = self._model()

    def _model(self):
        '''Constructs the full model network'''
        latent = self.encoder(self.input_layer)
        reconstructed = self.decoder(latent)
        model = Model(inputs=self.input_layer, outputs=reconstructed, name='AE')
        return model

    def call(self, inputs):
        return self.model(inputs)

    def fit(self, x, y=None, **kwargs):
        if y:
            return super().fit(x, y, **kwargs)
        else:
            return super().fit(x, x, **kwargs)

    def transform(self, inputs):
        return self.encoder.predict(inputs)
