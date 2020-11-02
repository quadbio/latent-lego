import numpy as np
import tensorflow as tf
from keras import backend as K

# Kernel functions slightly modified from https://github.com/theislab/scarches
def rbf_kernel(x, y):
    dim = K.cast(K.shape(x)[1], tf.float32)
    dist = squared_distance(x, y)
    return K.exp(-dist / dim)


def ms_rbf_kernel(x, y):
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
        1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]

    beta = 1. / (2. * (K.expand_dims(sigmas, 1)))
    dist = squared_distance(x, y)
    s = K.dot(beta, K.reshape(dist, (1, -1)))
    return K.reshape(tf.reduce_sum(tf.exp(-s), 0), K.shape(dist)) / len(sigmas)


def raphy_kernel(x, y, scales=[]):
    dist = K.expand_dims(squared_distance(x, y), 0)
    scales = K.expand_dims(K.expand_dims(scales, -1), -1)
    weights = K.eval(K.shape(scales)[0])
    weights = K.expand_dims(K.expand_dims(weights, -1), -1)
    return K.sum(weights * K.exp(-dist / (K.pow(scales, 2))), 0)


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
