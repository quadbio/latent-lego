"""Tensorflow Variational Autoencoder Models"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model

from .ae import Autoencoder, PoissonAutoencoder
from .ae import NegativeBinomialAutoencoder, ZINBAutoencoder
from .encoder import VariationalEncoder, TopologicalVariationalEncoder


class VariationalAutoencoder(Autoencoder):
    """Variational Autoencoder"""
    def __init__(
        self,
        kld_weight = 1e-5,
        prior = 'normal',
        iaf_units = [256, 256],
        n_pseudoinputs = 500,
        latent_dist = 'normal',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.kld_weight = tf.Variable(kld_weight, trainable=False)
        self.prior = prior
        self.iaf_units = iaf_units
        self.n_pseudoinputs = n_pseudoinputs
        self.latent_dist = latent_dist

        self.encoder = VariationalEncoder(
            kld_weight = self.kld_weight,
            prior = self.prior,
            iaf_units = self.iaf_units,
            n_pseudoinputs = self.n_pseudoinputs,
            latent_dim = self.latent_dim,
            dropout_rate = self.dropout_rate,
            batchnorm = self.batchnorm,
            l1 = self.l1,
            l2 = self.l2,
            activation = self.activation,
            initializer = self.initializer,
            hidden_units = self.hidden_units,
            latent_dist = self.latent_dist
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


class TopologicalVariationalAutoencoder(VariationalAutoencoder):
    """Variational autoencoder model with topological loss on latent space"""
    def __init__(self, topo_weight=1., **kwargs):
        super().__init__(**kwargs)
        self.topo_weight = topo_weight

        # Define components
        self.encoder = TopologicalVariationalEncoder(
            topo_weight = self.topo_weight,
            kld_weight = self.kld_weight,
            prior = self.prior,
            iaf_units = self.iaf_units,
            n_pseudoinputs = self.n_pseudoinputs,
            latent_dim = self.latent_dim,
            dropout_rate = self.dropout_rate,
            batchnorm = self.batchnorm,
            l1 = self.l1,
            l2 = self.l2,
            activation = self.activation,
            initializer = self.initializer,
            hidden_units = self.hidden_units
        )
