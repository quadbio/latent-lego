'''Tensorflow Autoencoder Model'''

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.losses import MeanSquaredError, Poisson

from .modules import Encoder
from .modules import Decoder, CountDecoder, PoissonDecoder, NegativeBinomialDecoder
from .losses import NegativeBinomial
from .layers import ColwiseMult, Slice

class Autoencoder(Model):
    def __init__(
        self,
        x_dim,
        latent_dim = 50,
        dropout_rate = 0.1,
        batchnorm = True,
        l1 = 0.0,
        l2 = 0.0,
        architecture = [128, 128],
        compile_model = True,
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

        self._encoder()
        self._decoder()

        if compile_model:
            self.compile()

    def _encoder(self):
        self.encoder = Encoder(
            x_dim = self.x_dim,
            latent_dim = self.latent_dim,
            dropout_rate = self.dropout_rate,
            batchnorm = self.batchnorm,
            l1 = self.l1,
            l2 = self.l2,
            architecture = self.architecture
        )

    def _decoder(self):
        self.decoder = Decoder(
            x_dim = self.x_dim,
            latent_dim = self.latent_dim,
            dropout_rate = self.dropout_rate,
            batchnorm = self.batchnorm,
            l1 = self.l1,
            l2 = self.l2,
            architecture = self.architecture[::-1]
        )

    def _loss(self):
        return MeanSquaredError()

    def call(self, inputs):
        '''Full forward pass through model'''
        latent = self.encoder(inputs)
        outputs = self.decoder(latent)
        return outputs

    def compile(self, optimizer='adam', loss=None, **kwargs):
        '''Compile model with default loss and omptimizer'''
        if not loss:
            loss = self._loss()
        return super().compile(loss=loss, optimizer=optimizer, **kwargs)

    def fit(self, x, y=None, **kwargs):
        if y:
            return super().fit(x, y, **kwargs)
        else:
            return super().fit(x, x, **kwargs)

    def transform(self, inputs):
        '''Map data (x) to latent space (z)'''
        return self.encoder.predict(inputs)



class CountAutoencoder(Autoencoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _decoder(self):
        self.decoder = CountDecoder(
            x_dim = self.x_dim,
            latent_dim = self.latent_dim,
            dropout_rate = self.dropout_rate,
            batchnorm = self.batchnorm,
            l1 = self.l1,
            l2 = self.l2,
            architecture = self.architecture[::-1]
        )

    def _loss(self):
        return MeanSquaredError()

    def call(self, inputs):
        '''Full forward pass through model'''
        x = inputs[0]
        sf = inputs[1]
        latent = self.encoder(x)
        outputs = self.decoder([latent, sf])
        return outputs

    def fit(self, x, y=None, **kwargs):
        if y:
            return super(Autoencoder, self).fit(x, y, **kwargs)
        else:
            return super(Autoencoder, self).fit(x, x[0], **kwargs)


class PoissonAutoencoder(CountAutoencoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _decoder(self):
        self.decoder = PoissonDecoder(
            x_dim = self.x_dim,
            latent_dim = self.latent_dim,
            dropout_rate = self.dropout_rate,
            batchnorm = self.batchnorm,
            l1 = self.l1,
            l2 = self.l2,
            architecture = self.architecture[::-1]
        )

    def _loss(self):
        return Poisson()


class NegativeBinomialAutoencoder(CountAutoencoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _decoder(self):
        self.decoder = NegativeBinomialDecoder(
            x_dim = self.x_dim,
            latent_dim = self.latent_dim,
            dropout_rate = self.dropout_rate,
            batchnorm = self.batchnorm,
            l1 = self.l1,
            l2 = self.l2,
            architecture = self.architecture[::-1]
        )

    def _loss(self):
        return None

    def call(self, inputs):
        '''Full forward pass through model'''
        x = inputs[0]
        sf = inputs[1]
        latent = self.encoder(x)
        outputs, disp = self.decoder([latent, sf])
        nb_loss = NegativeBinomial(theta=disp)
        self.add_loss(nb_loss(x, outputs))
        return outputs
