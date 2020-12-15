"""Tensorflow implementations of encoder models"""

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp
from typing import Iterable, Literal, Union, Callable

from latent.layers import DenseStack, PseudoInputs, DISTRIBUTIONS
from latent.losses import TopologicalSignatureDistance

tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors


class Encoder(keras.Model):
    """Encoder base class. This model compresses input data in a latent space
    with `latent_dim` dimensions by through passing it through a `DenseStack`.
    """
    def __init__(
        self,
        latent_dim: int = 50,
        name: str = 'encoder',
        initializer: Union[str, Callable] = 'glorot_normal',
        **kwargs
    ):
        """
        Arguments:
            latent_dim: Integer indicating the number of dimensions in the latent space.
            name: String indicating the name of the model.
            initializer: Initializer for the kernel weights matrix (see
                `keras.initializers`)
            **kwargs: Other arguments passed on to `DenseStack`
        """
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.initializer = keras.initializers.get(initializer)

        # Define components
        self.hidden_layers = DenseStack(
            name=self.name,
            initializer=self.initializer,
            **kwargs
        )
        self.final_layer = layers.Dense(
            self.latent_dim,
            name='encoder_latent',
            kernel_initializer=self.initializer,
            activation='linear'
        )

    def call(self, inputs):
        """Full forward pass through model"""
        h = self.hidden_layers(inputs)
        outputs = self.final_layer(h)
        return outputs


class TopologicalEncoder(Encoder):
    """Encoder model with topological regularization loss on latent space
    ([Moor 2019](https://arxiv.org/abs/1906.00722)).
    """
    def __init__(
        self,
        latent_dim: int = 50,
        name: str = 'topological_encoder',
        initializer: Union[str, Callable] = 'glorot_normal',
        topo_weight: float = 1.,
        **kwargs
    ):
        """
        Arguments:
            latent_dim: Integer indicating the number of dimensions in the latent space.
            name: String indicating the name of the model.
            initializer: Initializer for the kernel weights matrix (see
                `keras.initializers`)
            topo_weight: Float indicating the weight of the topological loss.
            **kwargs: Other arguments passed on to `DenseStack`
        """
        self.topo_weight = topo_weight
        self.topo_regularizer = TopologicalSignatureDistance()
        super().__init__(
            latent_dim=latent_dim,
            name=name,
            initializer=initializer,
            **kwargs
        )

    def call(self, inputs):
        """Full forward pass through model"""
        h = self.hidden_layers(inputs)
        outputs = self.final_layer(h)
        self.add_topo_loss(inputs, outputs)
        return outputs

    def add_topo_loss(self, inputs, outputs):
        """Added topological loss to final model"""
        topo_loss = self.topo_weight * self.topo_regularizer(inputs, outputs)
        self.add_loss(topo_loss)
        self.add_metric(topo_loss, name='topo_loss')


class VariationalEncoder(Encoder):
    """Variational encoder. This model compresses input data by parameterizing a latent
    distribution that is regularized through a KL Divergence loss."""
    def __init__(
        self,
        latent_dim: int = 50,
        name: str = 'variational_encoder',
        initializer: Union[str, Callable] = 'glorot_normal',
        kld_weight: float = 1e-4,
        prior: Literal['normal', 'iaf', 'vamp'] = 'normal',
        latent_dist: Literal['independent', 'multivariate'] = 'independent',
        iaf_units: Iterable[int] = [256, 256],
        n_pseudoinputs: int = 200,
        **kwargs
    ):
        """
        Arguments:
            latent_dim: Integer indicating the number of dimensions in the latent space.
            name: String indicating the name of the model.
            initializer: Initializer for the kernel weights matrix (see
                `keras.initializers`)
            kld_weight: Float indicating the weight of the KL Divergence
                regularization loss.
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
            **kwargs: Other arguments passed on to `DenseStack`.
        """
        self.kld_weight = tf.Variable(kld_weight, trainable=False)
        self.prior = prior
        self.iaf_units = iaf_units
        self.n_pseudoinputs = n_pseudoinputs
        self.latent_dist = latent_dist
        super().__init__(
            latent_dim=latent_dim,
            name=name,
            initializer=initializer,
            **kwargs
        )

        # Define components
        self.latent_dist_layer = DISTRIBUTIONS.get(latent_dist, tfpl.IndependentNormal)
        self.dist_param_layer = layers.Dense(
            self.latent_dist_layer.params_size(self.latent_dim),
            name='encoder_dist_params',
            kernel_initializer=self.initializer
        )
        self.sampling = self.latent_dist_layer(self.latent_dim)

        # Independent() reinterprets each latent_dim as an independent distribution
        self.prior_dist = tfd.Independent(
            tfd.Normal(loc=tf.zeros(self.latent_dim), scale=1.),
            reinterpreted_batch_ndims=1
        )

        if self.prior == 'iaf':
            # Inverse autoregressive flow (Kingma et al. 2016)
            made = tfb.AutoregressiveNetwork(params=2, hidden_units=self.iaf_units)
            self.prior_dist = tfd.TransformedDistribution(
                distribution=self.prior_dist,
                bijector=tfb.Invert(tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=made))
            )

        elif self.prior == 'vamp':
            # Variational mixture of posteriors (VAMP) prior (Tomczak & Welling 2018)
            self.pseudo_inputs = PseudoInputs(n_inputs=self.n_pseudoinputs)

    def call(self, inputs):
        """Full forward pass through model"""
        h = self.hidden_layers(inputs)
        dist_params = self.dist_param_layer(h)
        outputs = self.sampling(dist_params)
        self.add_kld_loss(inputs, outputs)
        return outputs

    def add_kld_loss(self, inputs, outputs):
        """Adds KLDivergence loss to model"""
        # VAMP prior depends on input, so we have to add it here
        if self.prior == 'vamp':
            prior_dist = self._vamp_prior(inputs)
            kld_loss = self.kld_weight * self._vamp_kld(outputs, prior_dist)
        else:
            prior_dist = self.prior_dist
            kld_regularizer = tfpl.KLDivergenceRegularizer(
                prior_dist,
                weight=self.kld_weight,
                test_points_reduce_axis=None
            )
            kld_loss = kld_regularizer(outputs)
        # Add losses manually to better monitor them
        self.add_loss(kld_loss)
        self.add_metric(kld_loss, name='kld_loss')

    def _vamp_prior(self, inputs):
        """Computes VAMP prior by feeding pseudoinputs through model"""
        # Inputs are needed to infer shape
        # and to ensure a connected graph
        h = self.pseudo_inputs(inputs)
        h = self.hidden_layers(h)
        dist_params = self.dist_param_layer(h)
        outputs = self.sampling(dist_params)
        return outputs

    # Adapted from original implementation https://github.com/jmtomczak/vae_vampprior
    def _vamp_kld(self, xdist, pdist):
        """Computes KLD between x and VAMP prior"""
        z = tf.convert_to_tensor(xdist)
        n_pseudo = tf.cast(self.n_pseudoinputs, tf.float32)
        x_log_prob = xdist.log_prob(z)
        zx = tf.expand_dims(z, 1)
        a = pdist.log_prob(zx) - tf.math.log(n_pseudo)
        a_max = tf.math.reduce_max(a, axis=1)
        p_log_prob = a_max + tf.math.reduce_logsumexp(a - tf.expand_dims(a_max, 1))
        # This can become negative, so we make sure we
        # actually just take the absolute difference
        kld = tf.math.reduce_mean(x_log_prob - p_log_prob)
        return tf.math.abs(kld)


class TopologicalVariationalEncoder(VariationalEncoder, TopologicalEncoder):
    """Variational encoder model with topological regularization loss on latent space
    ([Moor 2019](https://arxiv.org/abs/1906.00722)).
    """
    def __init__(
        self,
        latent_dim: int = 50,
        name: str = 'topo_variational_encoder',
        initializer: Union[str, Callable] = 'glorot_normal',
        kld_weight: float = 1e-4,
        prior: Literal['normal', 'iaf', 'vamp'] = 'normal',
        latent_dist: Literal['independent', 'multivariate'] = 'independent',
        iaf_units: Iterable[int] = [256, 256],
        n_pseudoinputs: int = 200,
        topo_weight: float = 1.,
        **kwargs
    ):
        """
        Arguments:
            latent_dim: Integer indicating the number of dimensions in the latent space.
            name: String indicating the name of the model.
            initializer: Initializer for the kernel weights matrix (see
                `keras.initializers`)
            kld_weight: Float indicating the weight of the KL Divergence
                regularization loss.
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
            topo_weight: Float indicating the weight of the topological loss.
            **kwargs: Other arguments passed on to `DenseStack`.
        """
        super().__init__(
            latent_dim=latent_dim,
            name=name,
            initializer=initializer,
            kld_weight=kld_weight,
            prior=prior,
            latent_dist=latent_dist,
            iaf_units=iaf_units,
            n_pseudoinputs=n_pseudoinputs,
            topo_weight=topo_weight,
            **kwargs
        )

    def call(self, inputs):
        """Full forward pass through model"""
        h = self.hidden_layers(inputs)
        dist_params = self.dist_param_layer(h)
        outputs = self.sampling(dist_params)
        self.add_kld_loss(inputs, outputs)
        self.add_topo_loss(inputs, outputs)
        return outputs
