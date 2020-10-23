import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda, Layer


class Slice(Layer):
    '''Slices inputs by index.'''
    def __init__(self, index=0, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, inputs, **kwargs):
        return inputs[self.index]


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
