import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1_l2
import tensorflow.keras.layers as layers

from .activations import ACTIVATIONS
from .losses import MaximumMeanDiscrepancy


### Core layers
class DenseBlock(layers.Layer):
    '''Basic dense layer block'''
    def __init__(
        self,
        units,
        name = 'dense_block',
        dropout_rate = 0.1,
        batchnorm = True,
        l1 = 0.0,
        l2 = 0.0,
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
        l1 = 0.0,
        l2 = 0.0,
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
    '''Uses inputs (z_mean, z_log_var) to sample z.'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class GradReversal(layers.Layer):
    '''Reverses gradient during backprop.'''
    def __init__(self, weight=1.0, **kwargs):
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


### Critic layers
class MMDCritic(layers.Layer):
    '''Adds MMD loss between conditions.'''
    def __init__(
        self,
        units,
        name = 'mmd_critic',
        weight = 1.0,
        n_conditions = 2,
        kernel_method = 'rbf',
        **kwargs
    ):
        super().__init__(name=name)
        self.units = units
        self.weight = weight
        self.n_conditions = n_conditions
        self.kernel_method = kernel_method

        # Define components
        self.mmd_layer = DenseBlock(
            units,
            dropout_rate = 0,
            **kwargs
        )

    def call(self, inputs):
        x, labels = inputs
        outputs = self.mmd_layer(x)
        mmd_loss = MaximumMeanDiscrepancy(
            n_conditions = self.n_conditions,
            kernel_method = self.kernel_method
        )
        crit_loss = self.weight * mmd_loss(labels, outputs)
        self.add_loss(crit_loss)
        self.add_metric(crit_loss, name=self.name)
        return outputs



class PairingCritic(layers.Layer):
    '''Matches paired points in latent space by forcing them to the same location.'''
    def __init__(
        self,
        units,
        name = 'pairing_critic',
        weight = 1.0,
        n_conditions = 2,
        kernel_method = 'rbf',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.units = units
        self.weight = weight

    def call(self, inputs):
        x1, x2  = inputs
        # Element-wise difference
        dist = tf.norm(tf.math.subtract(x1, x2), axis=0)
        crit_loss = self.weight * tf.math.reduce_sum(dist)
        self.add_loss(crit_loss)
        self.add_metric(crit_loss, name=self.name)
        return outputs


CRITICS = {
    'pairing': PairingCritic,
    'mmd': MMDCritic
}
