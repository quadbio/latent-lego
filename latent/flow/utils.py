import numpy as np
import tensorflow as tf
from keras import backend as K

# Kernel functions slightly modified from https://github.com/theislab/scarches
def rbf_kernel(x, y):
    dim = tf.cast(tf.shape(x)[1], tf.float32)
    dist = squared_distance(x, y)
    return tf.math.exp(-dist / dim)


def ms_rbf_kernel(x, y):
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
        1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]

    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    dist = squared_distance(x, y)
    s = K.dot(beta, tf.reshape(dist, (1, -1)))
    return tf.reshape(tf.math.reduce_sum(tf.exp(-s), 0), tf.shape(dist)) / len(sigmas)


def raphy_kernel(x, y, scales=[]):
    dist = tf.expand_dims(squared_distance(x, y), 0)
    scales = tf.expand_dims(tf.expand_dims(scales, -1), -1)
    weights = tf.eval(tf.shape(scales)[0])
    weights = tf.expand_dims(tf.expand_dims(weights, -1), -1)
    kernel = weights * tf.math.exp(-dist / (tf.math.pow(scales, 2)))
    return tf.math.reduce_sum(kernel, 0)


KERNELS = {
    'multiscale_rbf': ms_rbf_kernel,
    'rbf': rbf_kernel,
    'raphy': raphy_kernel
}


def squared_distance(x, y):
    r = tf.expand_dims(x, axis=1)
    return tf.math.reduce_sum(tf.math.square(r - y), axis=-1)


def nan2zero(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)


def nan2inf(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x) + np.inf, x)


def nelem(x):
    nelem = tf.math.reduce_sum(tf.cast(~tf.math.is_nan(x), tf.float32))
    return tf.cast(tf.where(tf.math.equal(nelem, 0.), 1., nelem), x.dtype)
