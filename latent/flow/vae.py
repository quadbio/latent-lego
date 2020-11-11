'''Tensorflow Variational Autoencoder Models'''

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model

from .ae import Autoencoder, PoissonAutoencoder
from .ae import NegativeBinomialAutoencoder, ZINBAutoencoder
from .encoder import VariationalEncoder


class VariationalAutoencoder(Autoencoder):
    '''Variational Autoencoder'''
    def __init__(
        self,
        kld_weight = 1e-5,
        prior = 'normal',
        iaf_units = [256, 256],
        vamp_pseudoinputs = 500,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.kld_weight = tf.Variable(kld_weight, trainable=False)
        self.prior = prior
        self.iaf_units = iaf_units
        self.vamp_pseudoinputs = vamp_pseudoinputs

        self.encoder = VariationalEncoder(
            kld_weight = self.kld_weight,
            prior = self.prior,
            iaf_units = self.iaf_units,
            vamp_pseudoinputs = self.vamp_pseudoinputs,
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
