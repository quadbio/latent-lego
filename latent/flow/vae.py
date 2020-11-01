'''Tensorflow Variational Autoencoder Models'''

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.losses import MeanSquaredError, Poisson

from .ae import Autoencoder, PoissonAutoencoder
from .ae import NegativeBinomialAutoencoder, ZINBAutoencoder
from .encoder import VariationalEncoder


class VariationalAutoencoder(Autoencoder):
    def __init__(self, beta=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

        self.encoder = VariationalEncoder(
            beta = self.beta,
            latent_dim = self.latent_dim,
            dropout_rate = self.dropout_rate,
            batchnorm = self.batchnorm,
            l1 = self.l1,
            l2 = self.l2,
            activation = self.activation,
            initializer = self.initializer,
            hidden_units = self.hidden_units
        )


class PoissonVAE(PoissonAutoencoder, VariationalAutoencoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class NegativeBinomialVAE(NegativeBinomialAutoencoder, VariationalAutoencoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ZINBVAE(ZINBAutoencoder, VariationalAutoencoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
