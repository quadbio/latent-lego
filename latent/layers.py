"""Tensorflow implementations of useful layers and blocks"""

import tensorflow as tf
import tensorflow.keras.initializers as initializers
import tensorflow.keras.activations as activations
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1_l2
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp
from typing import Iterable, Union, Callable
from ._compat import Literal

from .activations import ACTIVATIONS, clipped_exp
from .losses import MaximumMeanDiscrepancy, GromovWassersteinDistance
from .utils import log_density_gaussian, matrix_log_density_gaussian

tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors


# Core layers and stacks
class DenseBlock(layers.Layer):
    """Basic dense layer block with regularization, dropout, and batch-/layernorm
    functionality.
    """
    def __init__(
        self,
        units: int,
        name: str = None,
        dropout_rate: float = 0.1,
        batchnorm: bool = True,
        layernorm: bool = False,
        l1: float = 0.,
        l2: float = 0.,
        activation: Union[str, Callable] = 'leaky_relu',
        initializer: Union[str, Callable] = 'glorot_normal'
    ):
        """
        Arguments:
            units: Positive integer, dimensionality of the output space.
            name: String indicating the name of the layer.
            dropout_rate: Float between 0 and 1. Fraction of the input
                units to drop.
            batchnorm: Boolean, whether to perform batch normalization.
            layernorm: Boolean, whether to perform layer normalization.
            l1: Float. L1 regularization factor.
            l2: Float. L2 regularization factor.
            activation: Activation function to use.
            initializer: Initializer for the `kernel` weights matrix.
        """
        super().__init__(name=name)
        self.dropout_rate = dropout_rate
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.l1 = l1
        self.l2 = l2
        self.initializer = initializers.get(initializer)

        # Define block components
        self.dense = layers.Dense(
            units,
            kernel_initializer=self.initializer,
            kernel_regularizer=l1_l2(self.l1, self.l2)
        )
        self.bn = layers.BatchNormalization(center=True, scale=True)
        self.ln = layers.LayerNormalization(center=True, scale=True)
        if isinstance(activation, str):
            self.activation = ACTIVATIONS.get(activation, layers.LeakyReLU())
        else:
            self.activation = activation
        self.dropout = layers.Dropout(self.dropout_rate)

    # Full pass through the block
    def call(self, inputs, training=None):
        x = self.dense(inputs)
        if self.batchnorm:
            x = self.bn(x, training=training)
        if self.layernorm:
            x = self.ln(x)
        if self.dropout_rate > 0:
            x = self.dropout(x, training=training)
        x = self.activation(x)
        return x


class DenseStack(layers.Layer):
    """A stack of `DenseBlock` layers."""
    def __init__(
        self,
        name: str = None,
        hidden_units: Iterable[int] = [128, 128],
        conditional: Literal['first', 'all'] = None,
        **kwargs
    ):
        """
        Arguments:
            name: String indicating the name of the layer.
            hidden_units: Iterable of number hidden units per layer. All layers are fully
                connected. Ex. [128, 64] means first layer has 128 nodes and second one
                has 64.
            conditional:
                One of the following:\n
                * `'first'` Inject condition into first layer
                * `'all'` Inject condition into all layers
                * `None` Don't inject condition
            **kwargs: Other arguments passed on to `DenseBlock`.
        """
        super().__init__(name=name)
        self.hidden_units = hidden_units
        self.conditional = conditional

        # Define stack
        self.dense_stack = []
        for idx, units in enumerate(self.hidden_units):
            layer_name = f'{self.name}_{idx}'
            layer = DenseBlock(units, name=layer_name, **kwargs)
            self.dense_stack += [layer]

    def call(self, inputs, training=None):
        if self.conditional:
            x, *conditions = inputs
        else:
            x = inputs
        for idx, layer in enumerate(self.dense_stack):
            if self._inject_condition(idx):
                x = tf.concat([x, *conditions], axis=-1)
            x = layer(x, training=training)
        return x

    def _inject_condition(self, idx):
        """Checks if conditions should be injected into layer"""
        if not self.conditional:
            return False
        elif self.conditional == 'all':
            return True
        elif self.conditional == 'first':
            return idx == 0


# Utility layers
class RowwiseMult(layers.Layer):
    """Performs row-wise multiplication between input vectors."""
    def __init__(self, name: str = None):
        """
        Arguments:
            name: String indicating the name of the layer.
        """
        super().__init__(name=name)

    def call(self, inputs):
        return inputs[0] * K.reshape(inputs[1], (-1, 1))


class Sampling(layers.Layer):
    """Uses inputs (z_mean, log_var) to sample z."""
    def __init__(self, name: str = None):
        """
        Arguments:
            name: String indicating the name of the layer.
        """
        super().__init__(name=name)

    def call(self, inputs):
        mean, log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.math.exp(0.5 * log_var) * epsilon


class SharedDispersion(layers.Layer):
    """Layer to get shared dispersion estimates per gene."""
    def __init__(
        self,
        units: int,
        name: str = None,
        activation: Union[str, Callable] = 'clipped_exp',
        initializer: Union[str, Callable] = 'glorot_normal'
    ):
        """
        Arguments:
            units: Positive integer, dimensionality of the output space.
            name: String indicating the name of the layer.
            activation: Activation function to use.
            initializer: Initializer for the `kernel` weights matrix.
        """
        super().__init__(name=name)
        self.units = units
        self.initializer = initializer

        if isinstance(activation, str):
            self.activation = ACTIVATIONS.get(activation, clipped_exp)
        else:
            self.activation = activation

    def build(self, input_shape):
        self.disp = self.add_weight(
            name='dispersion',
            shape=(1, self.units),
            initializer=self.initializer
        )

    def call(self, inputs):
        x = tf.broadcast_to(self.disp, (tf.shape(inputs)[0], self.units))
        x = self.activation(x)
        return x


class Constant(layers.Layer):
    """Layer that outputs a constant value."""
    def __init__(
        self,
        units: int,
        constant: float = 1.,
        name: str = None,
        trainable: bool = True,
        activation: Union[str, Callable] = 'clipped_exp'
    ):
        """
        Arguments:
            units: Positive integer, dimensionality of the output space.
            name: String indicating the name of the layer.
            trainable: Boolean, whether to perform updates during
                training.
            activation: Activation function to use.
        """
        super().__init__(name=name)
        self.units = units
        self.const = tf.Variable(
            [[constant]], dtype=tf.float32, trainable=trainable)
        if isinstance(activation, str):
            self.activation = ACTIVATIONS.get(activation, clipped_exp)
        else:
            self.activation = activation

    def call(self, inputs):
        x = tf.broadcast_to(self.const, (tf.shape(inputs)[0], self.units))
        x = self.activation(x)
        return x


# Implementation adapted from https://github.com/theislab/sfaira/
class PseudoInputs(layers.Layer):
    """Creates trainable pseudo inputs
    ([Tomczak 2017](https://arxiv.org/abs/1705.07120)) based on input
    shapes.
    """
    def __init__(
        self,
        n_inputs: int,
        name: str = None,
        activation: Union[str, Callable] = 'relu',
        initializer: Union[str, Callable] = None
    ):
        """
        Arguments:
            n_inputs: Positive integer, number of pseudo-inputs.
            name: String indicating the name of the layer.
            activation: Activation function to use.
            initializer: Initializer for the `kernel` weights matrix.
        """
        super().__init__(name=name)
        self.n_inputs = n_inputs
        self.activation = activations.get(activation)
        self.cond_activation = activations.get('hard_sigmoid')
        if initializer:
            self.initializer = initializers.get(initializer)
        else:
            self.initializer = tf.random_normal_initializer(mean=-0.05, stddev=0.01)

    def build(self, input_shape):
        self.input_shape = input_shape
        # Conditional version of pseudoinputs if input_shape is a tuple
        # Extra pseudoinput with hard sigmoid activation for conditions
        if isinstance(input_shape, (list, tuple)):
            input_shape, *cond_shapes = input_shape
            c_shape = tf.math.reduce_sum([s[-1] for s in cond_shapes])
            self.c = self.add_weight(
                shape=(self.n_inputs, c_shape),
                initializer=self.initializer,
                dtype=tf.float32,
                name='u'
            )
            self.conditional = True
        else:
            self.conditional = False
        # Add weight for non-conditional inputs
        self.u = self.add_weight(
            shape=(self.n_inputs, input_shape[-1]),
            initializer=self.initializer,
            dtype=tf.float32,
            name='u'
        )
        self.built = True

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return self.activation(self.u), self.cond_activation(self.c)
        else:
            return self.activation(self.u)


# Critic layers
class MMDCritic(layers.Layer):
    """Adds MMD loss between conditions."""
    def __init__(
        self,
        name: str = None,
        weight: float = 1.,
        n_conditions: int = 2,
        n_groups: int = None,
        kernel_method: Union[Literal['rbf', 'ms_rbf', 'rq'], Callable] = 'ms_rbf',
        **kwargs
    ):
        super().__init__(name=name)
        self.weight = weight
        self.n_conditions = n_conditions
        self.kernel_method = kernel_method
        self.n_groups = n_groups
        self.loss_func = MaximumMeanDiscrepancy(
            n_conditions=self.n_conditions,
            kernel_method=self.kernel_method
        )

    def call(self, inputs):
        if self.n_groups:
            outputs, cond, groups = inputs
            cond_out = tf.dynamic_partition(outputs, groups, self.n_groups)
            lab_out = tf.dynamic_partition(cond, groups, self.n_groups)
            # Apply critic within group
            crit_loss = []
            for i in range(len(cond_out)):
                crit_loss += [self.loss_func(lab_out[i], cond_out[i])]
            crit_loss = self.weight * tf.math.reduce_mean(crit_loss)
        else:
            outputs, cond = inputs
            crit_loss = self.weight * self.loss_func(cond, outputs)
        self.add_loss(crit_loss)
        self.add_metric(crit_loss, name=f'{self.name}_loss')
        return outputs


class PairwiseDistCritic(layers.Layer):
    """Matches paired points in latent space by forcing them to the same location."""
    def __init__(
        self,
        name: str = None,
        weight: float = 1.,
        **kwargs
    ):
        super().__init__(name=name)
        self.weight = weight

    def call(self, inputs):
        x, labels = inputs
        x1, x2 = tf.dynamic_partition(x, labels, 2)
        # Element-wise euclidean distance
        dist = tf.norm(tf.math.subtract(x1, x2), axis=0)
        crit_loss = self.weight * tf.math.reduce_mean(dist)
        self.add_loss(crit_loss)
        self.add_metric(crit_loss, name=f'{self.name}_loss')
        return x


class GromovWassersteinCritic(layers.Layer):
    """Adds Gromov-Wasserstein loss between conditions."""
    def __init__(
        self,
        name: str = None,
        method: Literal['gw', 'entropic_gw'] = 'gw',
        weight: float = 1.,
        **kwargs
    ):
        super().__init__(name=name)
        self.weight = weight
        self.loss_func = GromovWassersteinDistance(method=method)

    def call(self, inputs):
        x, labels = inputs
        x1, x2 = tf.dynamic_partition(x, labels, 2)
        # Element-wise difference
        crit_loss = self.weight * self.loss_func(x1, x2)
        self.add_loss(crit_loss)
        self.add_metric(crit_loss, name=f'{self.name}_loss')
        return x


CRITICS = {
    'pairing': PairwiseDistCritic,
    'mmd': MMDCritic,
    'wasserstein': GromovWassersteinCritic
}


# Loss layers
class KLDivergenceAddLoss(layers.Layer):
    """Computes weighted KLD loss with optional capacity as proposed in
    ([Burgess 2018](https://arxiv.org/abs/1804.03599)) and
    ([Higgins 2017](https://openreview.net/forum?id=Sy2fzU9gl)).
    """
    def __init__(
        self,
        distribution_b,
        weight: float = 1.,
        capacity: float = 0.,
        **kwargs
    ):
        """
        Arguments:
            distribution_b: Distribution instance corresponding to b as KL[a,b]
            weight: Weight of the KLD loss.
            capacity: Capacity of the loss. Can be linearly increased using a scheduler
                callback.
            **kwargs: Other arguments passed to `keras.layers.Layer`.
        """
        super().__init__(**kwargs)
        self.weight = weight
        self.capacity = capacity
        self.kld_regularizer = tfpl.KLDivergenceRegularizer(
            distribution_b,
            weight=1.,
            test_points_reduce_axis=None
        )

    def call(self, distribution_a):
        """Calculates KLDivergence"""
        kld_loss = self.weight * tf.math.maximum(
            0., self.kld_regularizer(distribution_a) - self.capacity)
        return kld_loss


# Implementation adapted from
# * https://github.com/YannDubs/disentangling-vae
# * https://github.com/julian-carpenter/beta-TCVAE
# * https://github.com/rtqichen/beta-tcvae
class DecomposedKLDAddLoss(layers.Layer):
    """Computes decomposed KLD loss with weights for individual terms
    using minibatch weighted sampling or minibatch stratified sampling
    according to ([Chen 2019](https://arxiv.org/abs/1802.04942)).
    """
    def __init__(
        self,
        distribution_b,
        data_size: int = 1000,
        mi_weight: float = 1.,
        tc_weight: float = 1.,
        kl_weight: float = 1.,
        capacity: float = 0.,
        full_decompose: bool = False,
        **kwargs
    ):
        """
        Arguments:
            data_size: Number of data points in the training set.
            mi_weight: Weight of the Index-Code mutual information term.
            tc_weight: Weight of the total correlation term.
            kl_weight: Weight of the dimension-wise KL term.
            capacity: Capacity of the loss. Can be linearly increased using a scheduler
                callback.
            full_decompose: Whether to fully decompose the KLD as α*MI + ß*TC + γ*dwKL
                or only calculate TC and write loss as γ*KLD + (ß-1) * TC (default).
            kwargs: Other arguments passed to `keras.layers.Layer`.
        """
        super().__init__(**kwargs)
        self.data_size = data_size
        self.mi_weight = mi_weight
        self.tc_weight = tc_weight
        self.kl_weight = kl_weight
        self.capacity = capacity
        self.full_decompose = full_decompose

        if not self.full_decompose:
            self.kld_layer = KLDivergenceAddLoss(
                distribution_b,
                weight=self.kl_weight,
                capacity=self.capacity
            )

    def call(self, distribution_a):
        log_pz, log_qz, log_qz_prod, log_qz_cond_x = self._get_kld_components(
            distribution_a)

        # Compute TC term
        # TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = tf.reduce_mean(log_qz - log_qz_prod)

        if not self.full_decompose:
            kld_loss = self.kld_layer(distribution_a)
            return self.kl_weight * kld_loss + self.tc_weight * tc_loss
        else:
            # Compute other components of KLD loss
            # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
            mi_loss = tf.reduce_mean(log_qz_cond_x - log_qz)
            # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
            dw_kl_loss = tf.reduce_mean(log_qz_prod - log_pz)
            # Compute total KLD loss
            return (self.mi_weight * mi_loss
                + self.tc_weight * tc_loss
                + self.kl_weight * dw_kl_loss)

    def _get_kld_components(self, distribution_a):
        latent_sample = tf.convert_to_tensor(distribution_a)
        batch_size = tf.cast(tf.shape(latent_sample)[0], dtype=tf.float32)
        norm_const = tf.math.log(batch_size * self.data_size)
        loc = distribution_a.distribution.loc
        scale = distribution_a.distribution.scale

        # Calculate log p(z)
        # Zero mean and unit variance -> prior
        zeros = tf.zeros_like(latent_sample)
        log_pz = tf.reduce_sum(
            log_density_gaussian(latent_sample, zeros, 1), 1)

        # Calculate log q(z|x)
        log_qz_cond_x = tf.math.reduce_sum(
            log_density_gaussian(latent_sample, loc, scale), 1)

        log_qz_prob = matrix_log_density_gaussian(latent_sample, loc, scale)

        log_qz = tf.reduce_logsumexp(
            tf.reduce_sum(log_qz_prob, axis=2, keepdims=False),
            axis=1,
            keepdims=False
        ) - norm_const
        log_qz_prod = tf.reduce_sum(
            tf.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False) - norm_const,
            axis=1,
            keepdims=False
        )

        return log_pz, log_qz, log_qz_prod, log_qz_cond_x


# PROBABILISTIC LAYERS
DISTRIBUTIONS = {
    'independent': tfpl.IndependentNormal,
    'multivariate': tfpl.MultivariateNormalTriL
}
