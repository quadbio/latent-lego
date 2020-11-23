"""Tensorflow Variational Autoencoder Models"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model

from fastcore import delegates
from typing import Iterable, Literal, Union, Callable

from .ae import Autoencoder, PoissonAutoencoder
from .ae import NegativeBinomialAutoencoder, ZINBAutoencoder
from .encoder import VariationalEncoder, TopologicalVariationalEncoder


@delegates()
class VariationalAutoencoder(Autoencoder):
    """Variational Autoencoder"""
    def __init__(
        self,
        name: str = 'variational_autoencoder',
        kld_weight: float = 1e-5,
        prior: Literal['normal', 'iaf', 'vamp'] = 'normal',
        iaf_units: Iterable[int] = [256, 256],
        n_pseudoinputs: int = 200,
        latent_dist: Literal['independent', 'multivariate'] = 'independent',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
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
            hidden_units = self.encoder_units,
            latent_dist = self.latent_dist,
            **self.net_kwargs
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


@delegates()
class TopologicalVariationalAutoencoder(VariationalAutoencoder):
    """Variational autoencoder model with topological loss on latent space"""
    def __init__(self, topo_weight:float = 1., **kwargs):
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
            hidden_units = self.encoder_units,
            **self.net_kwargs
        )
