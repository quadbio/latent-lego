"""Tensorflow implementations of encoder models"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1_l2
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses

import tensorflow_probability as tfp
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors

from typing import Iterable, Literal, Union, Callable

from latent.activations import clipped_softplus, clipped_exp
from latent.layers import ColwiseMult, DenseStack, PseudoInputs, Sampling, DISTRIBUTIONS
from latent.losses import TopologicalSignatureDistance
from latent.utils import delegates


@delegates(DenseStack)
class Encoder(keras.Model):
    """Encoder base model"""
    def __init__(
        self,
        latent_dim: int = 50,
        name: str = 'encoder',
        initializer: Union[str, Callable] = 'glorot_normal',
        **kwargs
    ):
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.initializer = keras.initializers.get(initializer)

        # Define components
        self.hidden_layers = DenseStack(
            name = self.name,
            initializer = self.initializer,
            **kwargs
        )
        self.final_layer = layers.Dense(
            self.latent_dim,
            name = 'encoder_latent',
            kernel_initializer = self.initializer,
            activation = 'linear'
        )

    def call(self, inputs):
        """Full forward pass through model"""
        h = self.hidden_layers(inputs)
        outputs = self.final_layer(h)
        return outputs


@delegates()
class TopologicalEncoder(Encoder):
    """Encoder model with topological loss on latent space"""
    def __init__(
        self,
        name: str = 'topological_encoder',
        topo_weight: float = 1.,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.topo_weight = topo_weight
        self.topo_regularizer = TopologicalSignatureDistance()

    def call(self, inputs):
        """Full forward pass through model"""
        h = self.hidden_layers(inputs)
        outputs = self.final_layer(h)
        topo_loss = self.add_topo_loss(inputs, outputs)
        return outputs

    def add_topo_loss(self, inputs, outputs):
        """Added topological loss to final model"""
        topo_loss = self.topo_weight * self.topo_regularizer(inputs, outputs)
        self.add_loss(topo_loss)
        self.add_metric(topo_loss, name='topo_loss')


@delegates()
class VariationalEncoder(Encoder):
    """Variational encoder"""
    def __init__(
        self,
        name: str = 'variational_encoder',
        kld_weight: float = 1e-4,
        prior: Literal['normal', 'iaf', 'vamp'] = 'normal',
        latent_dist: Literal['independent', 'multivariate'] = 'independent',
        iaf_units: Iterable[int] = [256, 256],
        n_pseudoinputs: int = 200,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.kld_weight = tf.Variable(kld_weight, trainable=False)
        self.prior = prior
        self.iaf_units = iaf_units
        self.n_pseudoinputs = n_pseudoinputs
        self.latent_dist = latent_dist

        # Define components
        self.latent_dist_layer = DISTRIBUTIONS.get(latent_dist, tfpl.IndependentNormal)
        self.dist_param_layer = layers.Dense(
            self.latent_dist_layer.params_size(self.latent_dim),
            name = 'encoder_dist_params',
            kernel_initializer = self.initializer
        )
        self.sampling = self.latent_dist_layer(self.latent_dim)

        # Independent() reinterprets each latent_dim as an independent distribution
        self.prior_dist = tfd.Independent(
            tfd.Normal(loc=tf.zeros(self.latent_dim), scale=1.),
            reinterpreted_batch_ndims = 1
        )

        if self.prior == 'iaf':
            # Inverse autoregressive flow (Kingma et al. 2016)
            made = tfb.AutoregressiveNetwork(params=2, hidden_units=self.iaf_units)
            self.prior_dist = tfd.TransformedDistribution(
                distribution = self.prior_dist,
                bijector = tfb.Invert(tfb.MaskedAutoregressiveFlow(
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
                weight = self.kld_weight,
                test_points_reduce_axis = None
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
        h = self.dense_stack(h)
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


@delegates()
class TopologicalVariationalEncoder(VariationalEncoder, TopologicalEncoder):
    """Variational encoder model with topological loss on latent space"""
    def __init__(
        self,
        name: str = 'topological_variational_encoder',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

    def call(self, inputs):
        """Full forward pass through model"""
        h = self.hidden_layers(inputs)
        dist_params = self.dist_param_layer(h)
        outputs = self.sampling(dist_params)
        self.add_kld_loss(inputs, outputs)
        self.add_topo_loss(inputs, outputs)
        return outputs
