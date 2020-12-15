"""Tensorflow Variational Autoencoder Models"""

import tensorflow as tf
import tensorflow.keras as keras
from typing import Iterable, Literal, Callable, Union
from latent.modules import VariationalEncoder, TopologicalVariationalEncoder

from latent.modules import PoissonDecoder, NegativeBinomialDecoder, ZINBDecoder
from .ae import Autoencoder


class VariationalAutoencoder(Autoencoder):
    """Variational Autoencoder base class. This model uses a fixed variational encoder
    to fit a posterior distribution as the latent space trough regularization
    by a Kullback-Leibler Divergence loss.
    """
    def __init__(
        self,
        decoder: keras.Model = None,
        name: str = 'variational_autoencoder',
        x_dim: int = None,
        latent_dim: int = 50,
        encoder_units: Iterable[int] = [128, 128],
        decoder_units: Iterable[int] = [128, 128],
        reconstruction_loss: Callable = None,
        use_conditions: bool = False,
        kld_weight: float = 1e-5,
        prior: Literal['normal', 'iaf', 'vamp'] = 'normal',
        latent_dist: Literal['independent', 'multivariate'] = 'independent',
        iaf_units: Iterable[int] = [256, 256],
        n_pseudoinputs: int = 200,
        **kwargs
    ):
        """
        Arguments:
            decoder: Keras/tensorflow model object that inputs the latent space and
                outputs the reconstructed data. If not provided, a default model will be
                constructed from the arguments.
            name: String indicating the name of the model.
            x_dim: Integer indicating the number of features in the input data.
            latent_dim: Integer indicating the number of dimensions in the latent space.
            encoder_units: Integer list indicating the number of units of the encoder
                layers. Only used if `encoder` is not provided.
            decoder_units: An integer list indicating the number of units of the decoder
                layers. Only used if `decoder` is not provided.
            reconstruction_loss: Loss function applied to the reconstructed data and to be
                added by the decoder. Only used if `decoder` is not provided. Can also be
                added later by calling `compile()`.
            use_conditions: Boolean, whether to force the unpacking of conditions from the
                inputs.
            kld_weight: Float indicating the weight of the KL Divergence regularization
                loss.
            prior: The choice of prior distribution. One of the following:\n
                * `'normal'` - A unit gaussian (normal) distribution.
                * `'iaf'` - A unit gaussian with a Inverse Autoregressive Flows bijector
                    ([Kingma 2016](https://arxiv.org/abs/1606.04934))
                * `'vamp'` - A variational mixture of posteriors (VAMP) prior
                    ([Tomczak 2017](https://arxiv.org/abs/1705.07120))
            latent_dist: The choice of latent distribution. One of the following:\n
                * `'independent'` - A independent normal produced by
                    `tfpl.IndependentNormal`.
                * `'multivariate'` - A multivariate normal produced by
                    `tfpl.MultivariateNormalTriL`.
            iaf_units: Integer list indicating the units in the IAF bijector network.
                Only used if `prior = 'iaf'`.
            n_pseudoinputs: Integer indicating the number of pseudoinputs for the VAMP
                prior. Only used if `prior = 'vamp'`.
            **kwargs: Other arguments passed on to `DenseStack` for constructung encoder/
                decoder networks.
        """
        self.kld_weight = tf.Variable(kld_weight, trainable=False)
        self.prior = prior
        self.iaf_units = iaf_units
        self.n_pseudoinputs = n_pseudoinputs
        self.latent_dist = latent_dist
        var_encoder = VariationalEncoder(
            hidden_units=encoder_units,
            latent_dim=latent_dim,
            kld_weight=self.kld_weight,
            prior=self.prior,
            iaf_units=self.iaf_units,
            n_pseudoinputs=self.n_pseudoinputs,
            latent_dist=self.latent_dist,
            **kwargs
        )
        super().__init__(
            encoder=var_encoder,
            decoder=decoder,
            name=name,
            x_dim=x_dim,
            latent_dim=latent_dim,
            decoder_units=encoder_units,
            use_conditions=use_conditions,
            **kwargs
        )


class PoissonVAE(Autoencoder):
    """Poisson Variational Autoencoder with fixed variational encoder and poisson decoder
    networks.
    """
    def __init__(
        self,
        name: str = 'poisson_vae',
        x_dim: int = None,
        latent_dim: int = 50,
        encoder_units: Iterable[int] = [128, 128],
        decoder_units: Iterable[int] = [128, 128],
        reconstruction_loss: Callable = None,
        use_conditions: bool = False,
        kld_weight: float = 1e-5,
        prior: Literal['normal', 'iaf', 'vamp'] = 'normal',
        latent_dist: Literal['independent', 'multivariate'] = 'independent',
        iaf_units: Iterable[int] = [256, 256],
        n_pseudoinputs: int = 200,
        **kwargs
    ):
        """
        Arguments:
            decoder: Keras/tensorflow model object that inputs the latent space and
                outputs the reconstructed data. If not provided, a default model will be
                constructed from the arguments.
            name: String indicating the name of the model.
            x_dim: Integer indicating the number of features in the input data.
            latent_dim: Integer indicating the number of dimensions in the latent space.
            encoder_units: Integer list indicating the number of units of the encoder
                layers. Only used if `encoder` is not provided.
            decoder_units: An integer list indicating the number of units of the decoder
                layers. Only used if `decoder` is not provided.
            reconstruction_loss: Loss function applied to the reconstructed data and to be
                added by the decoder. Only used if `decoder` is not provided. Can also be
                added later by calling `compile()`.
            use_conditions: Boolean, whether to force the unpacking of conditions from the
                inputs.
            kld_weight: Float indicating the weight of the KL Divergence regularization
                loss.
            prior: The choice of prior distribution. One of the following:\n
                * `'normal'` - A unit gaussian (normal) distribution.
                * `'iaf'` - A unit gaussian with a Inverse Autoregressive Flows bijector
                    ([Kingma 2016](https://arxiv.org/abs/1606.04934))
                * `'vamp'` - A variational mixture of posteriors (VAMP) prior
                    ([Tomczak 2017](https://arxiv.org/abs/1705.07120))
            latent_dist: The choice of latent distribution. One of the following:\n
                * `'independent'` - A independent normal produced by
                    `tfpl.IndependentNormal`.
                * `'multivariate'` - A multivariate normal produced by
                    `tfpl.MultivariateNormalTriL`.
            iaf_units: Integer list indicating the units in the IAF bijector network.
                Only used if `prior = 'iaf'`.
            n_pseudoinputs: Integer indicating the number of pseudoinputs for the VAMP
                prior. Only used if `prior = 'vamp'`.
            **kwargs: Other arguments passed on to `DenseStack` for constructung encoder/
                decoder networks.
        """
        self.kld_weight = tf.Variable(kld_weight, trainable=False)
        self.prior = prior
        self.iaf_units = iaf_units
        self.n_pseudoinputs = n_pseudoinputs
        self.latent_dist = latent_dist
        poisson_decoder = PoissonDecoder(
            x_dim=x_dim,
            hidden_units=decoder_units,
            **kwargs
        )
        var_encoder = VariationalEncoder(
            hidden_units=encoder_units,
            latent_dim=latent_dim,
            kld_weight=self.kld_weight,
            prior=self.prior,
            iaf_units=self.iaf_units,
            n_pseudoinputs=self.n_pseudoinputs,
            latent_dist=self.latent_dist,
            **kwargs
        )
        super().__init__(
            encoder=var_encoder,
            decoder=poisson_decoder,
            name=name,
            x_dim=x_dim,
            latent_dim=latent_dim,
            use_conditions=use_conditions,
            **kwargs
        )


class NegativeBinomialVAE(Autoencoder):
    """Negative binomial variational autoencoder with fixed variational encoder and
    negative binomial decoder networks.
    """
    def __init__(
        self,
        name: str = 'nb_vae',
        x_dim: int = None,
        latent_dim: int = 50,
        encoder_units: Iterable[int] = [128, 128],
        decoder_units: Iterable[int] = [128, 128],
        reconstruction_loss: Callable = None,
        use_conditions: bool = False,
        dispersion: Union[Literal['gene', 'cell-gene', 'constant'], float] = 'constant',
        kld_weight: float = 1e-5,
        prior: Literal['normal', 'iaf', 'vamp'] = 'normal',
        latent_dist: Literal['independent', 'multivariate'] = 'independent',
        iaf_units: Iterable[int] = [256, 256],
        n_pseudoinputs: int = 200,
        **kwargs
    ):
        """
        Arguments:
            decoder: Keras/tensorflow model object that inputs the latent space and
                outputs the reconstructed data. If not provided, a default model will be
                constructed from the arguments.
            name: String indicating the name of the model.
            x_dim: Integer indicating the number of features in the input data.
            latent_dim: Integer indicating the number of dimensions in the latent space.
            encoder_units: Integer list indicating the number of units of the encoder
                layers. Only used if `encoder` is not provided.
            decoder_units: An integer list indicating the number of units of the decoder
                layers. Only used if `decoder` is not provided.
            reconstruction_loss: Loss function applied to the reconstructed data and to be
                added by the decoder. Only used if `decoder` is not provided. Can also be
                added later by calling `compile()`.
            use_conditions: Boolean, whether to force the unpacking of conditions from the
                inputs.
            dispersion:
                One of the following:\n
                * `'gene'` - dispersion parameter of NB is constant per gene across
                    cells
                * `'cell-gene'` - dispersion can differ for every gene in every cell
                * `'constant'` - dispersion is constant across all genes and cells
                * `float` - numeric value of fixed dispersion parameter
            kld_weight: Float indicating the weight of the KL Divergence regularization
                loss.
            prior: The choice of prior distribution. One of the following:\n
                * `'normal'` - A unit gaussian (normal) distribution.
                * `'iaf'` - A unit gaussian with a Inverse Autoregressive Flows bijector
                    ([Kingma 2016](https://arxiv.org/abs/1606.04934))
                * `'vamp'` - A variational mixture of posteriors (VAMP) prior
                    ([Tomczak 2017](https://arxiv.org/abs/1705.07120))
            latent_dist: The choice of latent distribution. One of the following:\n
                * `'independent'` - A independent normal produced by
                    `tfpl.IndependentNormal`.
                * `'multivariate'` - A multivariate normal produced by
                    `tfpl.MultivariateNormalTriL`.
            iaf_units: Integer list indicating the units in the IAF bijector network.
                Only used if `prior = 'iaf'`.
            n_pseudoinputs: Integer indicating the number of pseudoinputs for the VAMP
                prior. Only used if `prior = 'vamp'`.
            **kwargs: Other arguments passed on to `DenseStack` for constructung encoder/
                decoder networks.
        """
        self.dispersion = dispersion
        self.kld_weight = tf.Variable(kld_weight, trainable=False)
        self.prior = prior
        self.iaf_units = iaf_units
        self.n_pseudoinputs = n_pseudoinputs
        self.latent_dist = latent_dist
        nb_decoder = NegativeBinomialDecoder(
            x_dim=x_dim,
            hidden_units=decoder_units,
            dispersion=self.dispersion,
            **kwargs
        )
        var_encoder = VariationalEncoder(
            hidden_units=encoder_units,
            latent_dim=latent_dim,
            kld_weight=self.kld_weight,
            prior=self.prior,
            iaf_units=self.iaf_units,
            n_pseudoinputs=self.n_pseudoinputs,
            latent_dist=self.latent_dist,
            **kwargs
        )
        super().__init__(
            encoder=var_encoder,
            decoder=nb_decoder,
            name=name,
            x_dim=x_dim,
            latent_dim=latent_dim,
            use_conditions=use_conditions,
            **kwargs
        )


class ZINBVAE(Autoencoder):
    """Zero-inflated negative binomial variational autoencoder with fixed variational
    encoder and ZINB decoder networks.
    """
    def __init__(
        self,
        name: str = 'nb_vae',
        x_dim: int = None,
        latent_dim: int = 50,
        encoder_units: Iterable[int] = [128, 128],
        decoder_units: Iterable[int] = [128, 128],
        reconstruction_loss: Callable = None,
        use_conditions: bool = False,
        dispersion: Union[Literal['gene', 'cell-gene', 'constant'], float] = 'constant',
        kld_weight: float = 1e-5,
        prior: Literal['normal', 'iaf', 'vamp'] = 'normal',
        latent_dist: Literal['independent', 'multivariate'] = 'independent',
        iaf_units: Iterable[int] = [256, 256],
        n_pseudoinputs: int = 200,
        **kwargs
    ):
        """
        Arguments:
            decoder: Keras/tensorflow model object that inputs the latent space and
                outputs the reconstructed data. If not provided, a default model will be
                constructed from the arguments.
            name: String indicating the name of the model.
            x_dim: Integer indicating the number of features in the input data.
            latent_dim: Integer indicating the number of dimensions in the latent space.
            encoder_units: Integer list indicating the number of units of the encoder
                layers. Only used if `encoder` is not provided.
            decoder_units: An integer list indicating the number of units of the decoder
                layers. Only used if `decoder` is not provided.
            reconstruction_loss: Loss function applied to the reconstructed data and to be
                added by the decoder. Only used if `decoder` is not provided. Can also be
                added later by calling `compile()`.
            use_conditions: Boolean, whether to force the unpacking of conditions from the
                inputs.
            dispersion:
                One of the following:\n
                * `'gene'` - dispersion parameter of NB is constant per gene across
                    cells
                * `'cell-gene'` - dispersion can differ for every gene in every cell
                * `'constant'` - dispersion is constant across all genes and cells
                * `float` - numeric value of fixed dispersion parameter
            kld_weight: Float indicating the weight of the KL Divergence regularization
                loss.
            prior: The choice of prior distribution. One of the following:\n
                * `'normal'` - A unit gaussian (normal) distribution.
                * `'iaf'` - A unit gaussian with a Inverse Autoregressive Flows bijector
                    ([Kingma 2016](https://arxiv.org/abs/1606.04934))
                * `'vamp'` - A variational mixture of posteriors (VAMP) prior
                    ([Tomczak 2017](https://arxiv.org/abs/1705.07120))
            latent_dist: The choice of latent distribution. One of the following:\n
                * `'independent'` - A independent normal produced by
                    `tfpl.IndependentNormal`.
                * `'multivariate'` - A multivariate normal produced by
                    `tfpl.MultivariateNormalTriL`.
            iaf_units: Integer list indicating the units in the IAF bijector network.
                Only used if `prior = 'iaf'`.
            n_pseudoinputs: Integer indicating the number of pseudoinputs for the VAMP
                prior. Only used if `prior = 'vamp'`.
            **kwargs: Other arguments passed on to `DenseStack` for constructung encoder/
                decoder networks.
        """
        self.dispersion = dispersion
        self.kld_weight = tf.Variable(kld_weight, trainable=False)
        self.prior = prior
        self.iaf_units = iaf_units
        self.n_pseudoinputs = n_pseudoinputs
        self.latent_dist = latent_dist
        zinb_decoder = ZINBDecoder(
            x_dim=x_dim,
            hidden_units=decoder_units,
            dispersion=self.dispersion,
            **kwargs
        )
        var_encoder = VariationalEncoder(
            hidden_units=encoder_units,
            latent_dim=latent_dim,
            kld_weight=self.kld_weight,
            prior=self.prior,
            iaf_units=self.iaf_units,
            n_pseudoinputs=self.n_pseudoinputs,
            latent_dist=self.latent_dist,
            **kwargs
        )
        super().__init__(
            encoder=var_encoder,
            decoder=zinb_decoder,
            name=name,
            x_dim=x_dim,
            latent_dim=latent_dim,
            use_conditions=use_conditions,
            **kwargs
        )


class TopologicalVariationalAutoencoder(Autoencoder):
    """Variational autoencoder model with topological loss on latent space"""
    def __init__(
        self,
        decoder: keras.Model = None,
        name: str = 'variational_autoencoder',
        x_dim: int = None,
        latent_dim: int = 50,
        encoder_units: Iterable[int] = [128, 128],
        decoder_units: Iterable[int] = [128, 128],
        reconstruction_loss: Callable = None,
        use_conditions: bool = False,
        kld_weight: float = 1e-5,
        topo_weight: float = 1.,
        prior: Literal['normal', 'iaf', 'vamp'] = 'normal',
        latent_dist: Literal['independent', 'multivariate'] = 'independent',
        iaf_units: Iterable[int] = [256, 256],
        n_pseudoinputs: int = 200,
        **kwargs
    ):
        """
        Arguments:
            decoder: Keras/tensorflow model object that inputs the latent space and
                outputs the reconstructed data. If not provided, a default model will be
                constructed from the arguments.
            name: String indicating the name of the model.
            x_dim: Integer indicating the number of features in the input data.
            latent_dim: Integer indicating the number of dimensions in the latent space.
            encoder_units: Integer list indicating the number of units of the encoder
                layers. Only used if `encoder` is not provided.
            decoder_units: An integer list indicating the number of units of the decoder
                layers. Only used if `decoder` is not provided.
            reconstruction_loss: Loss function applied to the reconstructed data and to be
                added by the decoder. Only used if `decoder` is not provided. Can also be
                added later by calling `compile()`.
            use_conditions: Boolean, whether to force the unpacking of conditions from the
                inputs.
            kld_weight: Float indicating the weight of the KL Divergence regularization
                loss.
            topo_weight: Float indicating the weight of the topological loss.
            prior: The choice of prior distribution. One of the following:\n
                * `'normal'` - A unit gaussian (normal) distribution.
                * `'iaf'` - A unit gaussian with a Inverse Autoregressive Flows bijector
                    ([Kingma 2016](https://arxiv.org/abs/1606.04934))
                * `'vamp'` - A variational mixture of posteriors (VAMP) prior
                    ([Tomczak 2017](https://arxiv.org/abs/1705.07120))
            latent_dist: The choice of latent distribution. One of the following:\n
                * `'independent'` - A independent normal produced by
                    `tfpl.IndependentNormal`.
                * `'multivariate'` - A multivariate normal produced by
                    `tfpl.MultivariateNormalTriL`.
            iaf_units: Integer list indicating the units in the IAF bijector network.
                Only used if `prior = 'iaf'`.
            n_pseudoinputs: Integer indicating the number of pseudoinputs for the VAMP
                prior. Only used if `prior = 'vamp'`.
            **kwargs: Other arguments passed on to `DenseStack` for constructung encoder/
                decoder networks.
        """
        self.kld_weight = tf.Variable(kld_weight, trainable=False)
        self.topo_weight = tf.Variable(topo_weight, trainable=False)
        self.prior = prior
        self.iaf_units = iaf_units
        self.n_pseudoinputs = n_pseudoinputs
        self.latent_dist = latent_dist
        topo_var_encoder = TopologicalVariationalEncoder(
            latent_dim=latent_dim,
            hidden_units=encoder_units,
            topo_weight=self.topo_weight,
            kld_weight=self.kld_weight,
            prior=self.prior,
            iaf_units=self.iaf_units,
            n_pseudoinputs=self.n_pseudoinputs,
            **kwargs
        )
        super().__init__(
            encoder=topo_var_encoder,
            decoder=decoder,
            name=name,
            x_dim=x_dim,
            latent_dim=latent_dim,
            use_conditions=use_conditions,
            **kwargs
        )
