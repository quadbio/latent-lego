import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda, Layer


class ColwiseMult(Layer):
    '''Performs column-wise multiplication between input vectors.'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return inputs[0] * K.reshape(inputs[1], (-1, 1))


class Sampling(Layer):
    '''Uses inputs (z_mean, z_log_var) to sample z.'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class KLDivergenceAddLoss(Layer):
    """
    Identity transform layer that adds KL divergence
    to the final model loss.
    """
    def __init__(self, weight=1, **kwargs):
        self.weight = weight
        super().__init__(**kwargs)

    def call(self, inputs):
        mu, log_var = inputs
        kl_loss = - 0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
        self.add_loss(self.weight * tf.reduce_mean(kl_loss))
        return inputs
