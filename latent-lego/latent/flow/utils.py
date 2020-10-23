import numpy as np
import tensorflow as tf
from keras import backend as K

def nan2zero(x):
    return tf.where(tf.is_nan(x), tf.zeros_like(x), x)


def nan2inf(x):
    return tf.where(tf.is_nan(x), tf.zeros_like(x) + np.inf, x)


def nelem(x):
    nelem = tf.reduce_sum(tf.cast(~tf.is_nan(x), tf.float32))
    return tf.cast(tf.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)


def reduce_mean(x):
    nelem = nelem(x)
    x = nan2zero(x)
    return tf.divide(tf.reduce_sum(x), nelem)
