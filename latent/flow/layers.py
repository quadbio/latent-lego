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

from .activations import ACTIVATIONS
from .losses import MaximumMeanDiscrepancy


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
        self.initializer = keras.initializers.get(initializer)

        # Define block components
        self.dense = layers.Dense(
            units,
            name = self.name,
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
        mmd_loss = MaximumMeanDiscrepancy(
            n_conditions = self.n_conditions,
            kernel_method = self.kernel_method
        )
        crit_loss = self.weight * mmd_loss(labels, outputs)
        self.add_loss(crit_loss)
        self.add_metric(crit_loss, name=f'{self.name}_loss')
        return outputs


class PairwiseDistCritic(layers.Layer):
    '''Matches paired points in latent space by forcing them to the same location.'''
    def __init__(
        self,
        name = 'pairing_critic',
        weight = 1.,
        n_conditions = 2,
        hidden_units = None,
        kernel_method = 'rbf',
        **kwargs
    ):
        super().__init__(name=name)
        self.hidden_units = hidden_units
        self.weight = weight

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
        x1, x2 = tf.dynamic_partition(outputs, labels, 2)
        # Element-wise difference
        dist = tf.norm(tf.math.subtract(x1, x2), axis=0)
        crit_loss = self.weight * dist
        self.add_loss(crit_loss)
        self.add_metric(crit_loss, name=f'{self.name}_loss')
        return outputs


CRITICS = {
    'pairing': PairwiseDistCritic,
    'mmd': MMDCritic
}


### PROBABILISTIC LAYERS
class IndependentVonMisesFisher(tfpl.DistributionLambda):
    '''An independent VonMisesFisher distribution layer'''
    def __init__(
        self,
        event_size,
        convert_to_tensor_fn = tfd.Distribution.sample,
        validate_args = False,
        **kwargs
    ):
        super().__init__(
            lambda t: IndependentVonMisesFisher.new(t, event_size, validate_args),
            convert_to_tensor_fn, **kwargs
        )

        self._event_size = event_size
        self._convert_to_tensor_fn = convert_to_tensor_fn
        self._validate_args = validate_args

    @staticmethod
    def new(params, event_size, validate_args=False, name=None):
        '''Create the distribution instance from a `params` vector.'''
        params = tf.convert_to_tensor(params, name='params')
        event_size = tf.convert_to_tensor(event_size, dtype_hint=tf.int32)
        output_shape = tf.concat([
            tf.shape(params)[:-1],
            event_size,
        ], axis=0)
        mean_dir_params, conc_params = tf.split(params, 2, axis=-1)
        dist = tfd.Independent(
            tfd.VonMisesFisher(
                mean_direction = tf.reshape(mean_dir_params, output_shape),
                concentration = tf.math.softplus(tf.reshape(conc_params, output_shape)),
                validate_args = validate_args),
            reinterpreted_batch_ndims = tf.size(event_size),
            validate_args = validate_args
        )
        return dist

    @staticmethod
    def params_size(event_size):
        '''The number of `params` needed to create a single distribution.'''
        event_size = tf.convert_to_tensor(event_size, dtype_hint=tf.int32)
        event_size_const = tf.get_static_value(event_size)
        if event_size_const is not None:
            return 2 * np.prod(event_shape_const)
        else:
            return 2 * tf.reduce_prod(event_shape)


DISTRIBUTIONS = {
    'independent_normal': tfpl.IndependentNormal,
    'multivariate_normal': tfpl.MultivariateNormalTriL,
    'independent_vmf': IndependentVonMisesFisher
}
