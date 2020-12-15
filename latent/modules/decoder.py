"""Tensorflow implementations of decoder models"""

import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from typing import Iterable, Literal, Union, Callable

from latent.activations import clipped_exp
from latent.layers import ColwiseMult, DenseStack, SharedDispersion, Constant
from latent.losses import NegativeBinomial, ZINB
from latent.losses import get as get_loss


class Decoder(keras.Model):
    """Decoder base model. This model decompresses a latent space to reconstruct the
    input data by passing it through a `DenseStack`. It also takes care of adding the
    reconstruction loss to the model.
    """
    def __init__(
        self,
        x_dim: int,
        name: str = 'decoder',
        hidden_units: Iterable[int] = [128, 128],
        reconstruction_loss: Callable = None,
        loss_name: str = 'rec_loss',
        initializer: Union[str, Callable] = 'glorot_normal',
        **kwargs
    ):
        """
        Arguments:
            x_dim: Integer indicating the number of dimensions in the input data.
            name: String indicating the name of the model.
            hidden_units: Number of hidden units in `DenseStack`. If set to `None` the
                model skips the `DenseStack` and reduces to a linear decoder
                ([Svensson 2020](https://doi.org/10.1093/bioinformatics/btaa169)).
            reconstruction_loss: Function to compute reconstruction loss.
            loss_name: String indicating the name of the loss.
            initializer: Initializer for the kernel weights matrix (see
                `keras.initializers`)
            **kwargs: Other arguments passed on to `DenseStack`
        """
        super().__init__(name=name)
        self.x_dim = x_dim
        self.hidden_units = hidden_units
        self.reconstruction_loss = get_loss(reconstruction_loss)()
        self.loss_name = loss_name
        self.initializer = keras.initializers.get(initializer)
        # Set use_sf to False because this base model only expects one input
        self.use_sf = False

        # Define components
        if self.hidden_units:
            self.hidden_layers = DenseStack(
                name=f'{self.name}_hidden',
                hidden_units=self.hidden_units,
                initializer=self.initializer,
                **kwargs
            )
        self.final_layer = layers.Dense(
            self.x_dim,
            name=f'{self.name}_output',
            kernel_initializer=self.initializer,
            activation='linear'
        )

    def call(self, inputs):
        """Full forward pass through model"""
        x, latent = inputs
        h = self.hidden(latent)
        outputs = self.final_layer(h)
        self.add_reconstruction_loss(x, outputs)
        return outputs

    def hidden(self, latent):
        """Pass through hidden layers"""
        return self.hidden_layers(latent) if self.hidden_units else latent

    def add_reconstruction_loss(self, x, output):
        """Adds reconstruction loss to final model loss"""
        if self.reconstruction_loss:
            rec_loss = self.reconstruction_loss(x, output)
            self.add_loss(rec_loss)
            self.add_metric(rec_loss, name=self.loss_name)


class PoissonDecoder(Decoder):
    """Decoder with poisson reconstruction loss. Uses size factors to deal with count
    data.
    """
    def __init__(
        self,
        x_dim: int,
        name: str = 'poisson_decoder',
        hidden_units: Iterable[int] = [128, 128],
        reconstruction_loss: Union[Callable, str] = 'poisson',
        loss_name: str = 'poisson_loss',
        initializer: Union[str, Callable] = 'glorot_normal',
        **kwargs
    ):
        """
        Arguments:
            x_dim: Integer indicating the number of dimensions in the input data.
            name: String indicating the name of the model.
            hidden_units: Number of hidden units in `DenseStack`. If set to `None` the
                model skips the `DenseStack` and reduces to a linear decoder
                ([Svensson 2020](https://doi.org/10.1093/bioinformatics/btaa169)).
            reconstruction_loss: Function to compute reconstruction loss.
            loss_name: String indicating the name of the loss.
            initializer: Initializer for the kernel weights matrix (see
                `keras.initializers`)
            **kwargs: Other arguments passed on to `DenseStack`.
        """
        super().__init__(
            x_dim=x_dim,
            name=name,
            hidden_units=hidden_units,
            reconstruction_loss=reconstruction_loss,
            loss_name=loss_name,
            initializer=initializer,
            **kwargs
        )
        # Here use_sf becomes True
        self.use_sf = True
        # Define new components
        self.mean_layer = layers.Dense(
            self.x_dim,
            name='mean',
            activation=clipped_exp,
            kernel_initializer=self.initializer
        )
        self.norm_layer = ColwiseMult(name='output')

    def call(self, inputs):
        """Full forward pass through model"""
        x, latent, sf = inputs
        h = self.hidden(latent)
        mean = self.mean_layer(h)
        outputs = self.norm_layer([mean, sf])
        self.add_reconstruction_loss(x, outputs)
        return outputs


class NegativeBinomialDecoder(PoissonDecoder):
    """Decoder with negative binomial reconstruction loss. Uses size factors to deal with
    count data.
    """
    def __init__(
        self,
        x_dim: int,
        name: str = 'nb_decoder',
        loss_name: str = 'nb_loss',
        hidden_units: Iterable[int] = [128, 128],
        initializer: Union[str, Callable] = 'glorot_normal',
        dispersion: Union[Literal['gene', 'cell-gene', 'constant'], float] = 'gene',
        **kwargs
    ):
        """
        Arguments:
            x_dim: Integer indicating the number of dimensions in the input data.
            name: String indicating the name of the model.
            hidden_units: Number of hidden units in `DenseStack`. If set to `None` the
                model skips the `DenseStack` and reduces to a linear decoder
                ([Svensson 2020](https://doi.org/10.1093/bioinformatics/btaa169)).
            loss_name: String indicating the name of the loss.
            initializer: Initializer for the kernel weights matrix (see
                `keras.initializers`)
            dispersion:
                One of the following:\n
                * `'gene'` - dispersion parameter of NB is constant per gene across
                    cells
                * `'cell-gene'` - dispersion can differ for every gene in every cell
                * `'constant'` - dispersion is constant across all genes and cells
                * `float` - numeric value of fixed dispersion parameter
            **kwargs: Other arguments passed on to `DenseStack`.
        """
        super().__init__(
            x_dim=x_dim,
            name=name,
            hidden_units=hidden_units,
            loss_name=loss_name,
            initializer=initializer,
            **kwargs
        )
        self.dispersion = dispersion
        self.use_sf = True

        # Define new components
        self.mean_layer = layers.Dense(
            self.x_dim, name='mean',
            activation=clipped_exp,
            kernel_initializer=self.initializer
        )
        if dispersion == 'cell-gene':
            self.dispersion_layer = layers.Dense(
                self.x_dim,
                name='dispersion',
                activation=clipped_exp,
                kernel_initializer=self.initializer
            )
        elif dispersion == 'gene':
            self.dispersion_layer = SharedDispersion(
                self.x_dim,
                name='shared_dispersion',
                activation=clipped_exp,
                initializer=self.initializer
            )
        elif dispersion == 'constant':
            self.dispersion_layer = Constant(
                self.x_dim,
                trainable=True,
                name='constant_dispersion',
                activation=clipped_exp
            )
        elif isinstance(dispersion, float):
            self.dispersion_layer = Constant(
                self.x_dim,
                constant=self.dispersion,
                trainable=False,
                name='constant_dispersion',
                activation=clipped_exp
            )
        self.norm_layer = ColwiseMult()

    def call(self, inputs):
        """Full forward pass through model"""
        x, latent, sf = inputs
        h = self.hidden(latent)
        mean = self.mean_layer(h)
        outputs = self.norm_layer([mean, sf])
        disp = self.dispersion_layer(h)
        self.reconstruction_loss = NegativeBinomial(theta=disp)
        self.add_reconstruction_loss(x, outputs)
        return outputs


class ZINBDecoder(NegativeBinomialDecoder):
    """Decoder with ZINB reconstruction loss. Uses size factors to deal with
    count data.
    """
    def __init__(
        self,
        x_dim: int,
        name: str = 'zinb_decoder',
        loss_name: str = 'zinb_loss',
        hidden_units: Iterable[int] = [128, 128],
        initializer: Union[str, Callable] = 'glorot_normal',
        dispersion: Union[Literal['gene', 'cell-gene', 'constant'], float] = 'gene',
        **kwargs
    ):
        """
        Arguments:
            x_dim: Integer indicating the number of dimensions in the input data.
            name: String indicating the name of the model.
            hidden_units: Number of hidden units in `DenseStack`. If set to `None` the
                model skips the `DenseStack` and reduces to a linear decoder
                ([Svensson 2020](https://doi.org/10.1093/bioinformatics/btaa169))
            loss_name: String indicating the name of the loss.
            initializer: Initializer for the kernel weights matrix (see
                `keras.initializers`)
            dispersion:
                One of the following:\n
                * `'gene'` - dispersion parameter of NB is constant per gene across
                    cells
                * `'cell-gene'` - dispersion can differ for every gene in every cell
                * `'constant'` - dispersion is constant across all genes and cells
                * `float` - numeric value of fixed dispersion parameter
            **kwargs: Other arguments passed on to `DenseStack`.
        """
        super().__init__(
            x_dim=x_dim,
            name=name,
            hidden_units=hidden_units,
            loss_name=loss_name,
            initializer=initializer,
            dispersion=dispersion,
            **kwargs
        )
        self.use_sf = True

        # Define new components
        self.pi_layer = layers.Dense(
            self.x_dim,
            name='dropout_rate',
            activation='sigmoid',
            kernel_initializer=self.initializer
        )

    def call(self, inputs):
        """Full forward pass through model"""
        x, latent, sf = inputs
        h = self.hidden(latent)
        mean = self.mean_layer(h)
        outputs = self.norm_layer([mean, sf])
        disp = self.dispersion_layer(h)
        pi = self.pi_layer(h)
        self.reconstruction_loss = ZINB(theta=disp, pi=pi)
        self.add_reconstruction_loss(x, outputs)
        return outputs
