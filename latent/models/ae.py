"""Tensorflow Autoencoder Models"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
import tensorflow.keras.losses as losses

from typing import Iterable, Literal, Union, Callable

from latent.modules import Encoder, TopologicalEncoder
from latent.modules import Decoder, PoissonDecoder, NegativeBinomialDecoder, ZINBDecoder
from latent.layers import DenseBlock
from latent.losses import NegativeBinomial, ZINB


class Autoencoder(keras.Model):
    """Autoencoder base class"""
    def __init__(
        self,
        encoder: keras.Model = None,
        decoder: keras.Model = None,
        name: str = 'autoencoder',
        x_dim: int = None,
        latent_dim: int = 50,
        encoder_units: Iterable[int] = [128, 128],
        decoder_units: Iterable[int] = [128, 128],
        compile_model: bool = True,
        reconstruction_loss: Callable = None,
        **kwargs
    ):
        super().__init__(name=name)
        self.latent_dim = int(latent_dim)
        self.x_dim = int(x_dim)
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
        self.reconstruction_loss = reconstruction_loss
        self.net_kwargs = kwargs

        # Define components
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = Encoder(
                latent_dim = self.latent_dim,
                hidden_units = self.encoder_units,
                **kwargs
            )

        if decoder:
            self.decoder = decoder
        else:
            self.decoder = Decoder(
                x_dim = self.x_dim,
                hidden_units = self.decoder_units,
                reconstruction_loss = self.reconstruction_loss,
                **kwargs
            )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x, latent):
        return self.decoder([x, latent])

    def call(self, inputs):
        """Full forward pass through model"""
        latent = self.encode(inputs)
        outputs = self.decode(inputs, latent)
        return outputs

    def compile(self, optimizer='adam', loss=None, **kwargs):
        """Compile model with default loss and omptimizer"""
        return super().compile(loss=loss, optimizer=optimizer, **kwargs)

    def fit(self, x, y=None, **kwargs):
        if y:
            return super().fit(x, y, **kwargs)
        else:
            return super().fit(x, x, **kwargs)

    def transform(self, inputs):
        """Map data (x) to latent space (z)"""
        return self.encoder.predict(inputs)


class PoissonAutoencoder(Autoencoder):
    """Poisson autoencoder for count data"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.decoder = PoissonDecoder(
            x_dim = self.x_dim,
            hidden_units = self.decoder_units,
            **self.net_kwargs
        )

    def decode(self, x, latent, size_factors):
        output = self.decoder([x, latent, size_factors])
        return output

    def call(self, inputs):
        """Full forward pass through model"""
        x, sf = inputs
        latent = self.encode(x)
        outputs = self.decode(x, latent, sf)
        return outputs

    def fit(self, x, y=None, **kwargs):
        if y:
            return super(Autoencoder, self).fit(x, y, **kwargs)
        else:
            return super(Autoencoder, self).fit(x, x[0], **kwargs)


class NegativeBinomialAutoencoder(PoissonAutoencoder):
    """Autoencoder with negative binomial loss for count data"""
    def __init__(
        self,
        dispersion: Union[Literal['gene', 'cell-gene', 'constant'], float] = 'gene',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dispersion = dispersion

        self.decoder = NegativeBinomialDecoder(
            x_dim = self.x_dim,
            dispersion = self.dispersion,
            hidden_units = self.decoder_units,
            **self.net_kwargs
        )


class ZINBAutoencoder(PoissonAutoencoder):
    """Autoencoder with ZINB loss for count data"""
    def __init__(
        self,
        dispersion: Union[Literal['gene', 'cell-gene', 'constant'], float] = 'gene',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dispersion = dispersion

        self.decoder = ZINBDecoder(
            x_dim = self.x_dim,
            dispersion = self.dispersion,
            hidden_units = self.decoder_units,
            **self.net_kwargs
        )


class TopologicalAutoencoder(Autoencoder):
    """Autoencoder model with topological loss on latent space"""
    def __init__(self, topo_weight:float = 1., **kwargs):
        super().__init__(**kwargs)
        self.topo_weight = topo_weight

        # Define components
        self.encoder = TopologicalEncoder(
            topo_weight = self.topo_weight,
            latent_dim = self.latent_dim,
            hidden_units = self.decoder_units,
            **self.net_kwargs
        )


class PoissonTopoAE(PoissonAutoencoder, TopologicalAutoencoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class NegativeBinomialTopoAE(NegativeBinomialAutoencoder, TopologicalAutoencoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ZINBTopoAE(ZINBAutoencoder, TopologicalAutoencoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
