import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss

from .utils import nelem, nan2zero, nan2inf, reduce_mean


class NegativeBinomial(Loss):
    def __init__(self, theta, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = tf.cast(eps, tf.float32)
        self.theta = tf.cast(theta, tf.float32)

    def call(self, y_true, y_pred):
        '''Negative binomial loss (negative log likelihood)'''
        x = y_true
        mu = y_pred

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

        return r1 - r2


# NB and ZINB loss from https://github.com/theislab/dca
class NB(object):
    def __init__(
        self,
        theta = None,
        masking = False,
        scope = 'nbinom_loss/',
        scale_factor = 1.0
    ):

        # For numerical stability
        self.eps = 1e-8
        self.scale_factor = scale_factor
        self.scope = scope
        self.masking = masking
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps

        with tf.name_scope(self.scope):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32) * scale_factor

            if self.masking:
                nelem = nelem(y_true)
                y_true = nan2zero(y_true)

            # Clip theta
            theta = tf.minimum(self.theta, 1e6)

            t1 = tf.lgamma(theta + eps) + tf.lgamma(y_true + 1.0) - tf.lgamma(y_true + theta + eps)
            t2 = (theta + y_true) * tf.log(1.0 + (y_pred / (theta + eps))) + (
                    y_true * (tf.log(theta + eps) - tf.log(y_pred + eps)))
            final = t1 + t2

            final = nan2inf(final)

            if mean:
                if self.masking:
                    final = tf.divide(tf.reduce_sum(final), nelem)
                else:
                    final = tf.reduce_mean(final)
            else:
                final = tf.reduce_sum(final)

        return final


class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, scope='zinb_loss/', **kwargs):
        super().__init__(scope=scope, **kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps

        with tf.name_scope(self.scope):
            # reuse existing NB neg.log.lik.
            # mean is always False here, because everything is calculated
            # element-wise. we take the mean only in the end
            nb_case = super().loss(y_true, y_pred, mean=False) - tf.log(1.0 - self.pi + eps)

            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32) * scale_factor
            theta = tf.minimum(self.theta, 1e6)

            zero_nb = tf.pow(theta / (theta + y_pred + eps), theta)
            zero_case = -tf.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)
            result = tf.where(tf.less(y_true, 1e-8), zero_case, nb_case)
            ridge = self.ridge_lambda * tf.square(self.pi)
            result += ridge

            if mean:
                if self.masking:
                    result = _reduce_mean(result)
                else:
                    result = tf.reduce_mean(result)

            result = nan2inf(result)

        return result


def negbinom(disp, scale_factor=1.0, eta=1.0):
    def loss(y_true, y_pred):
        nb_obj = NB(theta=disp, masking=False, scale_factor=scale_factor)
        return eta * nb_obj.loss(y_true, y_pred, mean=True)
    return loss


def zinb(pi, disp, ridge=0.1, eta=1.0):
    def loss(y_true, y_pred):
        zinb_obj = ZINB(pi, theta=disp, ridge_lambda=ridge)
        return eta * zinb_obj.loss(y_true, y_pred)
    return loss
