import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras.losses as losses

from .utils import ms_rbf_kernel, rbf_kernel, nan2zero
from .utils import KERNELS


def maximum_mean_discrepancy(x, y, kernel_method='multiscale_rbf'):
    if isinstance(kernel_method, str):
        kernel = KERNELS.get(kernel_method, ms_rbf_kernel)
    else:
        kernel = kernel_method
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    x_kernel = tf.math.reduce_mean(kernel(x, x))
    y_kernel = tf.math.reduce_mean(kernel(y, y))
    xy_kernel = tf.math.reduce_mean(kernel(x, y))
    return x_kernel + y_kernel - 2 * xy_kernel


class MaximumMeanDiscrepancy(losses.Loss):
    '''MMD loss function between conditions'''
    def __init__(
        self,
        n_conditions = 2,
        kernel_method = 'multiscale_rbf',
        **kwargs
    ):
        super().__init__()
        self.n_conditions = n_conditions
        if isinstance(kernel_method, str):
            self.kernel = KERNELS.get(kernel_method, ms_rbf_kernel)
        else:
            self.kernel = kernel_method

    def call(self, y_true, y_pred):
        '''Calculated MMD between labels in y_pred space'''

        # No different conditions, no loss
        if self.n_conditions == 1:
            return tf.zeros(1)

        # Check if tuple or single tensor
        # This makes it more convenient to use in different contexts
        if isinstance(y_true, (list, tuple)):
            _, labels = y_true
        else:
            labels = y_true

        labels = tf.reshape(tf.cast(labels, tf.int32), (-1,))
        conditions = tf.dynamic_partition(
            y_pred, labels,
            num_partitions = self.n_conditions
        )
        result = []
        for i in range(len(conditions)):
            for j in range(i):
                res = maximum_mean_discrepancy(
                    conditions[i], conditions[j],
                    kernel_method = self.kernel
                )
                result.append(res)
        return tf.cast(result, tf.float32)


# Implementation adapted from https://github.com/theislab/dca
class NegativeBinomial(losses.Loss):
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


# Implementation adapted from https://github.com/theislab/dca
class ZINB(losses.Loss):
    '''Zero-inflated negative binomial loss'''
    def __init__(self, pi, theta, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = tf.cast(eps, tf.float32)
        self.theta = tf.cast(theta, tf.float32)
        self.pi = tf.cast(pi, tf.float32)

    def call(self, y_true, y_pred):
        '''Calculates negative log likelihood of the ZINB distribution'''
        x = tf.cast(y_true, tf.float32)
        mu = tf.cast(y_pred, tf.float32)
        nb_loss = NegativeBinomial(self.theta, eps=self.eps, reduction='none')

        case_nonzero = nb_loss(x, mu) - tf.math.log(1.0 - self.pi + self.eps)
        nb_zero = tf.math.pow(self.theta / (self.theta + mu), self.theta)
        case_zero = - tf.math.log(self.pi + ((1.0 - self.pi) * nb_zero) + self.eps)
        res = tf.where(tf.math.less(mu, self.eps), case_zero, case_nonzero)

        return res
