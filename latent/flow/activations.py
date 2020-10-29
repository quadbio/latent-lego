import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, Activation


def clipped_exp(x):
    return tf.clip_by_value(K.exp(x), 1e-5, 1e6)


def clipped_softplus(x):
    return tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)


ACTIVATIONS = {
    'prelu': PReLU(),
    'relu': ReLU(),
    'leaky_relu': LeakyReLU(),
    'selu': Activation('selu'),
    'linear': Activation('linear'),
    'sigmoid': Activation('sigmoid'),
    'clipped_exp': Activation(clipped_exp),
    'clipped_softplus': Activation(clipped_softplus)
}
