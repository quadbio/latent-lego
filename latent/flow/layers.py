'''Tensorflow implementations of useful layers and blocks'''

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

from collections.abc import Iterable

from .activations import ACTIVATIONS, clipped_exp
from .losses import MaximumMeanDiscrepancy, GromovWassersteinDistance


### Core layers and stacks
class DenseBlock(layers.Layer):
    '''Basic dense layer block'''
    def __init__(
        self,
        units,
        name = 'dense_block',
        dropout_rate = 0.1,
        batchnorm = True,
        l1 = 0.,
        l2 = 0.,
        activation = 'leaky_relu',
        initializer = 'glorot_normal',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.dropout_rate = dropout_rate
        self.batchnorm =  batchnorm
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
        if isinstance(activation, str):
            self.activation = ACTIVATIONS.get(activation, layers.LeakyReLU())
        else:
            self.activation = activation
        self.dropout = layers.Dropout(self.dropout_rate)


    def call(self, inputs):
        '''Full forward pass through model'''
        h = self.dense(inputs)
        h = self.bn(h)
        h = self.activation(h)
        outputs = self.dropout(h)
        return outputs


class DenseStack(layers.Layer):
    '''Core dense layer stack of encoders and decoders'''
    def __init__(
        self,
        name = 'dense_stack',
        dropout_rate = 0.1,
        batchnorm = True,
        l1 = 0.,
        l2 = 0.,
        hidden_units = [128, 128],
        activation = 'leaky_relu',
        initializer = 'glorot_normal',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.hidden_units =  hidden_units

        # Define stack
        self.dense_stack = []
        for idx, dim in enumerate(self.hidden_units):
            layer_name = f'{self.name}_{idx}'
            layer = DenseBlock(
                dim,
                name = layer_name,
                dropout_rate = dropout_rate,
                batchnorm = batchnorm,
                initializer = initializer,
                l1 = l1,
                l2 = l2
            )
            self.dense_stack.append(layer)

    def call(self, inputs):
        '''Full forward pass through model'''
        h = inputs
        for layer in self.dense_stack:
            h = layer(h)
        outputs = h
        return outputs


### Utility layers
class ColwiseMult(layers.Layer):
    '''Performs column-wise multiplication between input vectors.'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return inputs[0] * K.reshape(inputs[1], (-1, 1))


class Sampling(layers.Layer):
    '''Uses inputs (z_mean, log_var) to sample z.'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        mean, log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.math.exp(0.5 * log_var) * epsilon


class SharedDispersion(layers.Layer):
    '''Layer to get shared dispersion estimates per gene.'''
    def __init__(
        self,
        units,
        activation = 'clipped_exp',
        initializer = 'glorot_normal',
        name = 'shared_dispersion',
        **kwargs
    ):
        super().__init__(**kwargs)
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
    '''Layer that outputs a constant value.'''
    def __init__(
        self,
        units,
        constant = 1.,
        trainable = True,
        activation = 'clipped_exp',
        name = 'constant',
        **kwargs
    ):
        super().__init__(**kwargs)
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
    '''Creates trainable pseudo inputs'''
    def __init__(
        self,
        n_inputs,
        activation = 'hard_sigmoid',
        initializer = None,
        **kwargs
    ):
        super().__init__(**kwargs)
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
    '''Reverses gradient during backprop.'''
    def __init__(self, weight=1., **kwargs):
        super().__init__(**kwargs)
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
    '''
    Identity transform layer that adds analytic KL divergence
    (based on mean ans log_var) to the final model loss.
    '''

    def __init__(self, name='kld', weight=1., **kwargs):
        super().__init__(name=name, **kwargs)
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
    '''Adds MMD loss between conditions.'''
    def __init__(
        self,
        name = 'mmd_critic',
        weight = 1.,
        n_conditions = 2,
        hidden_units = None,
        kernel_method = 'rbf',
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
            self.mmd_layer = DenseStack(
                hidden_units = self.hidden_units,
                dropout_rate = 0.,
                **kwargs
            )

    def call(self, inputs):
        outputs, labels = inputs
        if self.hidden_units:
            outputs = self.mmd_layer(outputs)
        crit_loss = self.weight * self.loss_func(labels, outputs)
        self.add_loss(crit_loss)
        self.add_metric(crit_loss, name=f'{self.name}_loss')
        return outputs


class PairwiseDistCritic(layers.Layer):
    '''Matches paired points in latent space by forcing them to the same location.'''
    def __init__(
        self,
        name = 'pairing_critic',
        weight = 1.,
        hidden_units = None,
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
        # Element-wise difference
        dist = tf.norm(tf.math.subtract(x1, x2), axis=0)
        crit_loss = self.weight * dist
        self.add_loss(crit_loss)
        self.add_metric(crit_loss, name=f'{self.name}_loss')
        return outputs


class GromovWassersteinCritic(layers.Layer):
    '''Adds Gromov-Wasserstein loss between conditions.'''
    def __init__(
        self,
        name = 'wasserstein_critic',
        method = 'gw',
        weight = 1.,
        hidden_units = None,
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
    'independent_normal': tfpl.IndependentNormal,
    'multivariate_normal': tfpl.MultivariateNormalTriL
}
