"""Tensorflow implementations of useful layers and blocks"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.initializers as initializers
import tensorflow.keras.activations as activations
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1_l2
import tensorflow.keras.layers as layers

import tensorflow_probability as tfp
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors

from typing import Iterable, Literal, Union, Callable

from .activations import ACTIVATIONS, clipped_exp
from .losses import MaximumMeanDiscrepancy, GromovWassersteinDistance
from .utils import l2_norm


### Core layers and stacks
class DenseBlock(layers.Layer):
    """Basic dense layer block"""
    def __init__(
        self,
        units: int,
        name: str = 'dense_block',
        dropout_rate: float = 0.1,
        batchnorm: bool = True,
        layernorm: bool = False,
        l1: float = 0.,
        l2: float = 0.,
        activation: Union[str, Callable] = 'leaky_relu',
        initializer: Union[str, Callable] = 'glorot_normal'
    ):
        super().__init__(name=name)
        self.dropout_rate = dropout_rate
        self.batchnorm =  batchnorm
        self.layernorm =  layernorm
        self.l1 = l1
        self.l2 = l2
        self.initializer = initializers.get(initializer)

        # Define block components
        self.dense = layers.Dense(
            units,
            kernel_initializer = self.initializer,
            kernel_regularizer = l1_l2(self.l1, self.l2)
        )
        self.bn = layers.BatchNormalization(center=True, scale=True)
        self.ln = layers.LayerNormalization(center=True, scale=True)
        if isinstance(activation, str):
            self.activation = ACTIVATIONS.get(activation, layers.LeakyReLU())
        else:
            self.activation = activation
        self.dropout = layers.Dropout(self.dropout_rate)


    def call(self, inputs):
        """Full forward pass through model"""
        h = self.dense(inputs)
        if self.batchnorm:
            h = self.bn(h)
        if self.layernorm:
            h = self.ln(h)
        h = self.activation(h)
        outputs = self.dropout(h)
        return outputs


class DenseStack(layers.Layer):
    """Core dense layer stack of encoders and decoders"""
    def __init__(
        self,
        name: str = 'dense_stack',
        hidden_units: Iterable[int] = [128, 128],
        conditional: Literal['first', 'all'] = None,
        **kwargs
    ):
        super().__init__(name=name)
        self.hidden_units =  hidden_units
        self.conditional =  conditional

        # Define stack
        self.dense_stack = []
        for idx, units in enumerate(self.hidden_units):
            layer_name = f'{self.name}_{idx}'
            layer = DenseBlock(units, name=layer_name, **kwargs)
            self.dense_stack += [layer]

    def call(self, inputs):
        """Full forward pass through model"""
        if self.conditional:
            h, *conditions = inputs
        else:
            h = inputs
        for idx, layer in enumerate(self.dense_stack):
            if self._inject_condition(idx):
                h = tf.concat([h, *conditions], axis=-1)
            h = layer(h)
        outputs = h
        return outputs

    def _inject_condition(self, idx):
        """Checks if conditions should be injected into layer"""
        if not self.conditional:
            return False
        elif self.conditional == 'all':
            return True
        elif self.conditional == 'first':
            return idx == 0


### Utility layers
class ColwiseMult(layers.Layer):
    """Performs column-wise multiplication between input vectors."""
    def __init__(self, name:str = 'colwise_mult'):
        super().__init__(name=name)

    def call(self, inputs):
        return inputs[0] * K.reshape(inputs[1], (-1, 1))


class Sampling(layers.Layer):
    """Uses inputs (z_mean, log_var) to sample z."""
    def __init__(self, name:str = 'sampling'):
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
        name: str = 'shared_dispersion',
        activation: Union[str, Callable] = 'clipped_exp',
        initializer: Union[str, Callable] = 'glorot_normal'
    ):
        super().__init__(name=name)
        self.units = units
        self.initializer = initializer

        if isinstance(activation, str):
            self.activation = ACTIVATIONS.get(activation, clipped_exp)
        else:
            self.activation = activation

    def build(self, input_shape):
        self.disp = self.add_weight(
            name = 'dispersion',
            shape = (1, self.units),
            initializer = self.initializer
        )

    def call(self, inputs):
        h = tf.broadcast_to(self.disp, (tf.shape(inputs)[0], self.units))
        outputs = self.activation(h)
        return outputs


class Constant(layers.Layer):
    """Layer that outputs a constant value."""
    def __init__(
        self,
        units: int,
        name: str = 'constant',
        constant: float = 1.,
        trainable: bool = True,
        activation: Union[str, Callable] = 'clipped_exp'
    ):
        super().__init__(name=name)
        self.units = units
        self.const = tf.Variable(
            [[constant]], dtype=tf.float32, trainable=trainable)
        if isinstance(activation, str):
            self.activation = ACTIVATIONS.get(activation, clipped_exp)
        else:
            self.activation = activation

    def call(self, inputs):
        h = tf.broadcast_to(self.const, (tf.shape(inputs)[0], self.units))
        outputs = self.activation(h)
        return outputs


# Implementation adapted from https://github.com/theislab/sfaira/
class PseudoInputs(layers.Layer):
    """Creates trainable pseudo inputs"""
    def __init__(
        self,
        n_inputs: int,
        name: str = 'pseudo_inputs',
        activation: Union[str, Callable] = 'hard_sigmoid',
        initializer: Union[str, Callable] = None
    ):
        super().__init__(name=name)
        self.n_inputs = n_inputs
        self.activation = activations.get(activation)
        if initializer:
            self.initializer = initializers.get(initializer)
        else:
            self.initializer = tf.random_normal_initializer(mean=-0.05, stddev=0.01)

    def build(self, input_shape):
        self.u = self.add_weight(
            shape = (self.n_inputs, input_shape[-1]),
            initializer = self.initializer,
            dtype = tf.float32,
            name = 'u'
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        return self.activation(self.u)


class GradReversal(layers.Layer):
    """Reverses gradient during backprop."""
    def __init__(self, name:str = 'grad_reversal', weight:float = 1.):
        super().__init__(name=name)
        self.weight = weight

    def call(self, inputs):
        return self.grad_reverse(inputs)

    @tf.custom_gradient
    def grad_reverse(self, x):
        y = tf.identity(x)
        def grad(dy):
            return -dy * self.weight
        return y, custom_grad


class KLDivergenceAddLoss(layers.Layer):
    """
    Identity transform layer that adds analytic KL divergence
    (based on mean ans log_var) to the final model loss.
    """

    def __init__(self, name: str = 'kld', weight: float = 1.):
        super().__init__(name=name)
        self.weight = weight

    def call(self, inputs):
        mean, log_var = inputs
        kl_loss = - 0.5 * tf.math.reduce_sum(
            1 + log_var - tf.math.square(mean) - tf.math.exp(log_var)
        )
        self.add_loss(self.weight * kl_loss)
        self.add_metric(self.weight * kl_loss, name=f'{self.name}_loss')
        return inputs


### Critic layers
class MMDCritic(layers.Layer):
    """Adds MMD loss between conditions."""
    def __init__(
        self,
        name: str = 'mmd_critic',
        weight: float = 1.,
        n_conditions: int = 2,
        hidden_units: Iterable[int] = None,
        kernel_method: Union[Literal['rbf', 'ms_rbf', 'rq'], Callable] = 'ms_rbf',
        **kwargs
    ):
        super().__init__(name=name)
        self.hidden_units = hidden_units
        self.weight = weight
        self.n_conditions = n_conditions
        self.kernel_method = kernel_method
        self.loss_func = MaximumMeanDiscrepancy(
            n_conditions = self.n_conditions,
            kernel_method = self.kernel_method
        )

        # Define components
        if self.hidden_units:
            self.hidden_layers = DenseStack(
                hidden_units = self.hidden_units,
                dropout_rate = 0.,
                **kwargs
            )

    def call(self, inputs):
        outputs, labels = inputs
        if self.hidden_units:
            outputs = self.hidden_layers(outputs)
        crit_loss = self.weight * self.loss_func(labels, outputs)
        self.add_loss(crit_loss)
        self.add_metric(crit_loss, name=f'{self.name}_loss')
        return outputs


class PairwiseDistCritic(layers.Layer):
    """Matches paired points in latent space by forcing them to the same location."""
    def __init__(
        self,
        name: str = 'pairing_critic',
        weight: float = 1.,
        hidden_units: Iterable[int] = None,
        **kwargs
    ):
        super().__init__(name=name)
        self.hidden_units = hidden_units
        self.weight = weight

        # Define components
        if self.hidden_units:
            self.hidden_layer = DenseStack(
                hidden_units = self.hidden_units,
                dropout_rate = 0.,
                **kwargs
            )

    def call(self, inputs):
        outputs, labels = inputs
        if self.hidden_units:
            outputs = self.hidden_layer(outputs)
        x1, x2 = tf.dynamic_partition(outputs, labels, 2)
        # Element-wise euclidean distance
        dist = tf.norm(tf.math.subtract(x1, x2), axis=0)
        crit_loss = self.weight * tf.math.reduce_mean(dist)
        self.add_loss(crit_loss)
        self.add_metric(crit_loss, name=f'{self.name}_loss')
        return outputs


class GromovWassersteinCritic(layers.Layer):
    """Adds Gromov-Wasserstein loss between conditions."""
    def __init__(
        self,
        name: str = 'wasserstein_critic',
        method: Literal['gw', 'entropic_gw'] = 'gw',
        weight: float = 1.,
        hidden_units: Iterable[int] = None,
        **kwargs
    ):
        super().__init__(name=name)
        self.hidden_units = hidden_units
        self.weight = weight
        self.loss_func = GromovWassersteinDistance(
            method = method
        )

        # Define components
        if self.hidden_units:
            self.hidden_layer = DenseStack(
                hidden_units = self.hidden_units,
                dropout_rate = 0.,
                **kwargs
            )

    def call(self, inputs):
        outputs, labels = inputs
        if self.hidden_units:
            outputs = self.hidden_layer(outputs)
        x1, x2 = tf.dynamic_partition(outputs, labels, 2)
        # Element-wise difference
        crit_loss = self.weight * self.loss_func(x1, x2)
        self.add_loss(crit_loss)
        self.add_metric(crit_loss, name=f'{self.name}_loss')
        return outputs


CRITICS = {
    'pairing': PairwiseDistCritic,
    'mmd': MMDCritic,
    'wasserstein': GromovWassersteinCritic
}


### PROBABILISTIC LAYERS
DISTRIBUTIONS = {
    'independent': tfpl.IndependentNormal,
    'multivariate': tfpl.MultivariateNormalTriL
}
