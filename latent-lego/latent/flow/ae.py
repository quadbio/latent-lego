'''Tensorflow Autoencoder Model'''

import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.losses import MeanSquaredError, Poisson

from .modules import Encoder, Decoder, CountDecoder, PoissonDecoder
from .losses import NegativeBinomial

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

        self.input_layer = Input(shape=(self.x_dim, ), name='data')
        self.encoder = self._encoder()
        self.decoder = self._decoder()
        inputs, outputs = self._build_model()
        self.model = Model(inputs=inputs, outputs=outputs, name='autoencoder')

        if compile_model:
            self.compile()

    def _encoder(self):
        encoder = Encoder(
            x_dim = self.x_dim,
            latent_dim = self.latent_dim,
            dropout_rate = self.dropout_rate,
            batchnorm = self.batchnorm,
            l1 = self.l1,
            l2 = self.l2,
            architecture = self.architecture
        )
        return encoder

    def _decoder(self):
        decoder = Decoder(
            x_dim = self.x_dim,
            latent_dim = self.latent_dim,
            dropout_rate = self.dropout_rate,
            batchnorm = self.batchnorm,
            l1 = self.l1,
            l2 = self.l2,
            architecture = self.architecture[::-1]
        )
        return decoder

    def _build_model(self):
        '''Constructs the full model network'''
        latent = self.encoder(self.input_layer)
        outputs = self.decoder(latent)
        return self.input_layer, outputs

    def _loss(self):
        return MeanSquaredError()

    def call(self, inputs):
        return self.model(inputs)

    def compile(self, optimizer='adam', loss=None, **kwargs):
        if not loss:
            loss = self._loss()
        return super().compile(loss=loss, optimizer=optimizer, **kwargs)

    def fit(self, x, y=None, **kwargs):
        if y:
            return super().fit(x, y, **kwargs)
        else:
            return super().fit(x, x, **kwargs)

    def transform(self, inputs):
        return self.encoder.predict(inputs)



class CountAutoencoder(Autoencoder):
    def __init__(self, **kwargs):
        self.sf_layer = Input(shape=(1, ), name='size_factors')
        super().__init__(**kwargs)

    def _decoder(self):
        decoder = CountDecoder(
            x_dim = self.x_dim,
            latent_dim = self.latent_dim,
            dropout_rate = self.dropout_rate,
            batchnorm = self.batchnorm,
            l1 = self.l1,
            l2 = self.l2,
            architecture = self.architecture[::-1]
        )
        return decoder

    def _build_model(self):
        '''Constructs the full model network'''
        latent = self.encoder(self.input_layer)
        outputs = self.decoder([latent, self.sf_layer])
        return [self.input_layer, self.sf_layer], outputs

    def _loss(self):
        return mse()

    def fit(self, x, y=None, **kwargs):
        if y:
            return super(Autoencoder, self).fit(x, y, **kwargs)
        else:
            return super(Autoencoder, self).fit(x, x[0], **kwargs)


class PoissonAutoencoder(CountAutoencoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _decoder(self):
        decoder = PoissonDecoder(
            x_dim = self.x_dim,
            latent_dim = self.latent_dim,
            dropout_rate = self.dropout_rate,
            batchnorm = self.batchnorm,
            l1 = self.l1,
            l2 = self.l2,
            architecture = self.architecture[::-1]
        )
        return decoder

    def _loss(self):
        return Poisson()


class NegativeBinomialAutoencoder(CountAutoencoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _decoder(self):
        decoder = NegativeBinomialDecoder(
            x_dim = self.x_dim,
            latent_dim = self.latent_dim,
            dropout_rate = self.dropout_rate,
            batchnorm = self.batchnorm,
            l1 = self.l1,
            l2 = self.l2,
            architecture = self.architecture[::-1]
        )
        return decoder

    def _loss(self):
        return NegativeBinomial()
