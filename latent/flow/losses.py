import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss

from .utils import ms_rbf_kernel, rbf_kernel, raphy_kernel
from .utils import KERNELS


def maximum_mean_discrepancy(self, x, y, kernel_method='multiscale_rbf'):
    if isinstance(kernel_method, str):
        self.kernel = KERNELS.get(kernel_method, ms_rbf_kernel)
    else:
        self.kernel = kernel_method
    x_kernel = kernel(x, x)
    y_kernel = kernel(y, y)
    xy_kernel = kernel(x, y)
    return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)


class MaximumMeanDiscrepancy(Loss):
    '''MMD loss function between conditions'''
    def __init__(
        self,
        n_conditions = 2,
        kernel_method = 'multiscale_rbf',
        weight = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_conditions = n_conditions
        self.weight = weight
        if isinstance(kernel_method, str):
            self.kernel = KERNELS.get(kernel_method, ms_rbf_kernel)
        else:
            self.kernel = kernel_method

    def call(self, y_true, y_pred):
        '''Calculated MMD between labels in y_pred space'''
        _, labels = y_true
        labels = K.reshape(K.cast(labels, 'int32'), (-1,))
        conditions = tf.dynamic_partition(
            y_pred, labels,
            num_partitions = self.n_conditions
        )
        loss = []
        for i in range(len(conditions)):
            for j in range(i):
                loss += maximum_mean_discrepancy(
                    conditions[i], conditions[j],
                    kernel_method = self.kernel
                )
        if n_conditions == 1:
            loss = 0
        loss = K.reshape(loss, (self.n_conditions, self.n_conditions))
        return self.weight * loss


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
        nb_loss = NegativeBinomial(self.theta, reduction='none')

        case_nonzero = nb_loss(x, mu) - tf.math.log(1.0 - self.pi + self.eps)
        nb_zero = tf.math.pow(self.theta / (self.theta + mu), self.theta)
        case_zero = - tf.math.log(self.pi + ((1.0 - self.pi) * nb_zero) + self.eps)
        res = tf.where(tf.math.less(mu, self.eps), case_zero, case_nonzero)

        return res
