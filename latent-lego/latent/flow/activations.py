import tensorflow as tf
from tensorflow.keras import backend as K


def clipped_exp(x):
    return tf.clip_by_value(K.exp(x), 1e-5, 1e6)


def clipped_softplus(x):
    return tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)
