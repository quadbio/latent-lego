'''Tensorflow Autoencoder Models'''

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.losses import MeanSquaredError, Poisson

from .encoder import Encoder
from .decoder import Decoder, CountDecoder, PoissonDecoder, NegativeBinomialDecoder
from .decoder import ZINBDecoder
from .losses import NegativeBinomial, ZINB


class Autoencoder(Model):
    '''Classical autoencoder'''
    def __init__(
        self,
        x_dim,
        latent_dim = 50,
        dropout_rate = 0.1,
        batchnorm = True,
        l1 = 0.0,
        l2 = 0.0,
        hidden_units = [128, 128],
        compile_model = True,
        activation = 'prelu',
        initializer = 'glorot_normal',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.latent_dim = int(latent_dim)
        self.x_dim = int(x_dim)
        self.dropout_rate = dropout_rate
        self.batchnorm =  batchnorm
        self.l1 = l1
        self.l2 = l2
        self.activation = activation
        self.hidden_units = hidden_units
        self.initializer = keras.initializers.get(initializer)

        # Define components
        self.encoder = Encoder(
            latent_dim = self.latent_dim,
            dropout_rate = self.dropout_rate,
            batchnorm = self.batchnorm,
            l1 = self.l1,
            l2 = self.l2,
            activation = self.activation,
            initializer = self.initializer,
            hidden_units = self.hidden_units
        )

        self.decoder = Decoder(
            x_dim = self.x_dim,
            dropout_rate = self.dropout_rate,
            batchnorm = self.batchnorm,
            l1 = self.l1,
            l2 = self.l2,
            activation = self.activation,
            initializer = self.initializer,
            hidden_units = self.hidden_units[::-1]
        )

        self.rec_loss = MeanSquaredError()

    def call(self, inputs):
        '''Full forward pass through model'''
        latent = self.encoder(inputs)
        outputs = self.decoder(latent)
        return outputs

    def compile(self, optimizer='adam', loss=None, **kwargs):
        '''Compile model with default loss and omptimizer'''
        if not loss:
            loss = self.rec_loss
        return super().compile(loss=loss, optimizer=optimizer, **kwargs)

    def fit(self, x, y=None, **kwargs):
        if y:
            return super().fit(x, y, **kwargs)
        else:
            return super().fit(x, x, **kwargs)

    def transform(self, inputs):
        '''Map data (x) to latent space (z)'''
        return self.encoder.predict(inputs)



class PoissonAutoencoder(Autoencoder):
    '''Normalizing autoencoder for count data'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.decoder = CountDecoder(
            x_dim = self.x_dim,
            dropout_rate = self.dropout_rate,
            batchnorm = self.batchnorm,
            l1 = self.l1,
            l2 = self.l2,
            activation = self.activation,
            initializer = self.initializer,
            hidden_units = self.hidden_units[::-1]
        )

        self.rec_loss = Poisson()

    def call(self, inputs):
        '''Full forward pass through model'''
        x, sf = inputs
        latent = self.encoder(x)
        outputs = self.decoder([latent, sf])
        return outputs

    def fit(self, x, y=None, **kwargs):
        if y:
            return super(Autoencoder, self).fit(x, y, **kwargs)
        else:
            return super(Autoencoder, self).fit(x, x[0], **kwargs)


class NegativeBinomialAutoencoder(CountAutoencoder):
    '''Autoencoder with negative binomial loss for count data'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.decoder = NegativeBinomialDecoder(
            x_dim = self.x_dim,
            dropout_rate = self.dropout_rate,
            batchnorm = self.batchnorm,
            l1 = self.l1,
            l2 = self.l2,
            activation = self.activation,
            initializer = self.initializer,
            hidden_units = self.hidden_units[::-1]
        )

        # Loss is added in call()
        self.rec_loss = None

    def call(self, inputs):
        '''Full forward pass through model'''
        x, sf = inputs
        latent = self.encoder(x)
        outputs, disp = self.decoder([latent, sf])
        # Add loss here so it can be parameterized by theta
        rec_loss = NegativeBinomial(theta=disp)
        self.add_loss(rec_loss(x, outputs))
        return outputs


class ZINBAutoencoder(CountAutoencoder):
    '''Autoencoder with ZINB loss for count data'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.decoder = ZINBDecoder(
            x_dim = self.x_dim,
            dropout_rate = self.dropout_rate,
            batchnorm = self.batchnorm,
            l1 = self.l1,
            l2 = self.l2,
            activation = self.activation,
            initializer = self.initializer,
            hidden_units = self.hidden_units[::-1]
        )

        # Loss is added in call()
        self.rec_loss = None

    def call(self, inputs):
        '''Full forward pass through model'''
        x, sf = inputs
        latent = self.encoder(x)
        outputs, disp, pi = self.decoder([latent, sf])
        # Add loss here so it can be parameterized by theta and pi
        nb_loss = ZINB(theta=disp, pi=pi)
        self.add_loss(nb_loss(x, outputs))
        return outputs
