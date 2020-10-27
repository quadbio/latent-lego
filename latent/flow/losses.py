import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss
from tensorflow.keras.losses.Reduction import NONE

from .utils import nelem, nan2zero, nan2inf, reduce_mean


class NegativeBinomial(Loss):
    '''Negative binomial loss'''
    def __init__(self, theta, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = tf.cast(eps, tf.float32)
        self.theta = tf.cast(theta, tf.float32)

    def call(self, y_true, y_pred):
        '''Calculates negative log likelihood of the NB distribution'''
        x = tf.cast(y_true, tf.float32)
        mu = tf.cast(y_pred, tf.float32)

        r1 = (
            tf.math.lgamma(self.theta)
            + tf.math.lgamma(x + 1)
            - tf.math.lgamma(x + self.theta)
        )
        log_theta_mu_eps = tf.math.log(self.theta + mu + self.eps)
        r2 = (
            self.theta * (tf.math.log(self.theta + self.eps) - log_theta_mu_eps)
            + x * (tf.math.log(mu + self.eps) - log_theta_mu_eps)
        )
        res = r1 - r2
        return res


class ZINB(NegativeBinomial):
    '''Zero-inflated negative binomial loss'''
    def __init__(self, pi, **kwargs):
        super().__init__(**kwargs)
        self.pi = tf.cast(pi, tf.float32)

    def call(self, y_true, y_pred):
        '''Calculates negative log likelihood of the ZINB distribution'''
        x = tf.cast(y_true, tf.float32)
        mu = tf.cast(y_pred, tf.float32)
        nb_loss = NegativeBinomial(self.theta, reduction=NONE)

        case_nonzero = nb_loss(x, mu) - tf.math.log(1.0 - self.pi + self.eps)
        nb_zero = tf.math.pow(self.theta / (self.theta + mu), self.theta)
        case_zero = - tf.math.log(self.pi + ((1.0 - self.pi) * nb_zero) + self.eps)
        res = tf.where(tf.math.less(mu, self.eps), case_zero, case_nonzero)

        return res
