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

    def encode(self, inputs):
        """Prepare input for encoder and encode"""
        if self._conditional_encoder():
            return self.encoder([inputs['x'], *inputs['cond']])
        else:
            return self.encoder(inputs)

    def decode(self, inputs, latent):
        """Prepare input for decoder and decode"""
        if self._use_sf() and not self._conditional_decoder():
            return self.decoder([inputs['x'], latent, inputs['sf']])
        if not self._use_sf() and not self._conditional_decoder():
            return self.decoder([inputs['x'], latent, inputs['sf']])
        if self._use_sf() and self._conditional_decoder():
            latent = [latent, *inputs['cond']]
            return self.decoder([inputs['x'], latent, inputs['sf']])
        if not self._use_sf() and self._conditional_decoder():
            latent = [latent, *inputs['cond']]
            return self.decoder([inputs['x'], latent])

    def call(self, inputs):
        """Full forward pass through model"""
        inputs = self.unpack_inputs(inputs)
        latent = self.encode(inputs)
        outputs = self.decode(inputs, latent)
        return outputs

    def compile(self, optimizer='adam', loss=None, **kwargs):
        """Compile model with default loss and optimizer"""
        return super().compile(loss=loss, optimizer=optimizer, **kwargs)

    def fit(self, x, y=None, **kwargs):
        if not y:
            y = x[0] if self._use_sf() else x
        return super().fit(x, y, **kwargs)

    def transform(self, inputs):
        """Map data (x) to latent space (z)"""
        return self.encoder.predict(inputs)

    def unpack_inputs(self, inputs):
        """Unpacks inputs into x, conditions and size_factors."""
        if self._use_sf() and self._use_conditions():
            x, *cond, sf = inputs
            return {'x': x, 'cond': cond, 'sf': sf}
        if self._use_sf() and not self._use_conditions():
            x, sf = inputs
            return {'x': x, 'sf': sf}
        if not self._use_sf() and not self._use_conditions():
            x = inputs
            return {'x': x}
        if not self._use_sf() and self._use_conditions():
            x, *cond = inputs
            return {'x': x, 'cond': cond}

    def _use_sf(self):
        """Determine whether decoder uses size factors"""
        if hasattr(self.decoder, 'use_sf'):
            return self.decoder.use_sf
        else:
            return False

    def _conditional_encoder(self):
        """Determine whether encoder injects conditions"""
        if hasattr(self.encoder, 'hidden_layers'):
            return self.encoder.hidden_layers.conditional != None
        else:
            return False

    def _conditional_decoder(self):
        """Determine whether decoder injects conditions"""
        if hasattr(self.decoder, 'hidden_layers'):
            return self.decoder.hidden_layers.conditional != None
        else:
            return False

    def _use_conditions(self):
        """Determine whether to use conditions in model"""
        return self._conditional_decoder() or self._conditional_encoder()


class PoissonAutoencoder(Autoencoder):
    """Poisson autoencoder for count data"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoder = PoissonDecoder(
            x_dim = self.x_dim,
            hidden_units = self.decoder_units,
            **self.net_kwargs
        )


class NegativeBinomialAutoencoder(Autoencoder):
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


class ZINBAutoencoder(Autoencoder):
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
