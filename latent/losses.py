"""Tensorflow implementations of losses for autoencoders"""

import tensorflow as tf
import tensorflow.keras.losses as losses
from typing import Union, Callable
from ._compat import Literal

from .utils import ms_rbf_kernel, persistent_homology, slice_matrix
from .utils import l2_norm, nan2zero, KERNELS, OT_DIST


def maximum_mean_discrepancy(x, y, kernel=ms_rbf_kernel):
    """Calculates maximum mean discrepancy."""
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    x_kernel = tf.math.reduce_mean(kernel(x, x))
    y_kernel = tf.math.reduce_mean(kernel(y, y))
    xy_kernel = tf.math.reduce_mean(kernel(x, y))
    return x_kernel + y_kernel - 2 * xy_kernel


class MaximumMeanDiscrepancy(losses.Loss):
    """Computes Maximum Mean Discrepancy (MMD) loss between conditions (`y_true`)
    in `y_pred`.
    """
    def __init__(
        self,
        n_conditions: int = 2,
        kernel_method: str = 'ms_rbf',
        **kwargs
    ):
        """
        Arguments:
            n_conditions: Positive integer indicating number of conditions.
            kernel_method:
                Name of kernel method to use. Can be one of the following:\n
                * `'ms_rbf'` Multi-scale RBF kernel
                    ([Lotfollahi 2019](https://arxiv.org/abs/1910.01791))
                * `'rbf'` Basic RBF kernel
                * `'rq'` Rational Quadratic kernel
            **kwargs: Other arguments passed to `keras.losses.Loss`.
        """
        super().__init__(**kwargs)
        self.n_conditions = n_conditions
        if isinstance(kernel_method, str):
            self.kernel = KERNELS.get(kernel_method, ms_rbf_kernel)
        else:
            self.kernel = kernel_method

    def call(self, y_true, y_pred):
        """Calculated MMD between labels in y_pred space"""

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
            num_partitions=self.n_conditions
        )
        result = []
        for i in range(len(conditions)):
            for j in range(i):
                res = maximum_mean_discrepancy(
                    conditions[i], conditions[j],
                    kernel=self.kernel
                )
                result.append(res)
        # Empty conditions will produce nan values
        return nan2zero(tf.cast(result, tf.float32))


# Implementation adapted from https://github.com/theislab/dca
class NegativeBinomial(losses.Loss):
    """Computes negative binomial loss between `y_true` and `y_pred` given a dispersion
    parameter (`theta`).
    """
    def __init__(self, theta: Union[tf.Tensor, float], eps: float = 1e-8, **kwargs):
        """
        Arguments:
            theta: Positive float. Dispersion parameter.
            eps: Positive float. Clipping value for numerical stability.
            **kwargs: Other arguments passed to `keras.losses.Loss`.
        """
        super().__init__(**kwargs)
        self.eps = tf.cast(eps, tf.float32)
        self.theta = tf.cast(theta, tf.float32)

    def call(self, y_true, y_pred):
        """Calculates negative log likelihood of the NB distribution"""
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
    """Computes zero-inflated negative binomial loss between `y_true` and `y_pred` given
    a dispersion parameter (`theta`) and dropout rate (`pi`).
    """
    def __init__(
        self,
        pi: Union[tf.Tensor, float],
        theta: Union[tf.Tensor, float],
        eps: float = 1e-8,
        **kwargs
    ):
        """
        Arguments:
            theta: Positive float. Dispersion parameter.
            pi: Positive float between 0 and 1. Dropout rate.
            eps: Positive float. Clipping value for numerical stability.
            **kwargs: Other arguments passed to `keras.losses.Loss`.
        """
        super().__init__(**kwargs)
        self.eps = tf.cast(eps, tf.float32)
        self.theta = tf.cast(theta, tf.float32)
        self.pi = tf.cast(pi, tf.float32)

    def call(self, y_true, y_pred):
        """Calculates negative log likelihood of the ZINB distribution"""
        x = tf.cast(y_true, tf.float32)
        mu = tf.cast(y_pred, tf.float32)
        nb_loss = NegativeBinomial(self.theta, eps=self.eps, reduction='none')

        case_nonzero = nb_loss(x, mu) - tf.math.log(1.0 - self.pi + self.eps)
        nb_zero = tf.math.pow(self.theta / (self.theta + mu), self.theta)
        case_zero = - tf.math.log(self.pi + ((1.0 - self.pi) * nb_zero) + self.eps)
        res = tf.where(tf.math.less(mu, self.eps), case_zero, case_nonzero)

        return res


# Implementation adapted from https://github.com/BorgwardtLab/topological-autoencoders
class TopologicalSignatureDistance(losses.Loss):
    """Computes distance between topological signatures
    ([Moor 2019](https://arxiv.org/abs/1906.00722)).
    """
    def __init__(
        self,
        match_edges: Literal['symmetric', 'random'] = None,
        eps: float = 1e-8,
        return_additional_metrics: bool = False,
        **kwargs
    ):
        """
        Arguments:
            match_edges:
                One of the following:\n
                * `'symmetric'` Match edged between signatures symmetrically
                * `'random'` Match edged between signatures randomly
                * `None` Don't match edges
            eps: Positive float. Clipping value for numerical stability.
            return_additional_metrics: Boolean, whether to return additional metrics.
            **kwargs: Other arguments passed to `keras.losses.Loss`.
        """
        super().__init__(**kwargs)
        self.match_edges = match_edges
        self.return_additional_metrics = return_additional_metrics
        self.eps = eps

    @staticmethod
    def _get_pairings(distances):
        return persistent_homology(tf.stop_gradient(distances))

    @staticmethod
    def _select_distances_from_pairs(distance_matrix, pairs):
        # Utility func slice_matrix because tf does not allow us to slice matrices
        selected_distances = slice_matrix(distance_matrix, pairs[:, 0], pairs[:, 1])
        return selected_distances

    @staticmethod
    def _sig_error(signature1, signature2):
        """Compute distance between two topological signatures."""
        return tf.reduce_sum(tf.square(signature1 - signature2), axis=-1)

    def _compute_distance_matrix(self, x):
        x_flat = tf.reshape(x, [tf.shape(x)[0], -1])
        # Custom l2_norm because tf.norm can cause problems with gradients
        distances = l2_norm(x_flat[:, None] - x_flat, axis=2, eps=self.eps)
        return distances

    def call(self, y_true, y_pred):
        """Return topological distance of two data spaces.

        Args:
            y_true: Coordinates in space 1 (x)
            y_pred: Coordinates in space 2 (latent)

        Returns:
            distance, [dict(additional outputs)]
        """
        distances1 = self._compute_distance_matrix(y_true)
        distances1 = distances1 / tf.math.reduce_max(distances1)
        distances2 = self._compute_distance_matrix(y_pred)
        distances2 = distances2 / tf.math.reduce_max(distances2)

        pairs1 = self._get_pairings(distances1)
        pairs2 = self._get_pairings(distances2)

        distance_components = {}

        if not self.match_edges:
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            distance = self._sig_error(sig1, sig2)

        elif self.match_edges == 'symmetric':
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            # Selected pairs of 1 on distances of 2 and vice versa
            sig1_2 = self._select_distances_from_pairs(distances2, pairs1)
            sig2_1 = self._select_distances_from_pairs(distances1, pairs2)

            distance1_2 = self._sig_error(sig1, sig1_2)
            distance2_1 = self._sig_error(sig2, sig2_1)

            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1

        elif self.match_edges == 'random':
            # Create random selection in order to verify if what we are seeing
            # is the topological constraint or an implicit latent space prior
            # for compactness
            n_instances = len(pairs1[0])
            pairs1 = tf.concat([
                tf.random.shuffle(tf.range(n_instances))[:, None],
                tf.random.shuffle(tf.range(n_instances))[:, None]
            ], axis=1)
            pairs2 = tf.concat([
                tf.random.shuffle(tf.range(n_instances))[:, None],
                tf.random.shuffle(tf.range(n_instances))[:, None]
            ], axis=1)

            sig1_1 = self._select_distances_from_pairs(
                distances1, (pairs1, None))
            sig1_2 = self._select_distances_from_pairs(
                distances2, (pairs1, None))

            sig2_2 = self._select_distances_from_pairs(
                distances2, (pairs2, None))
            sig2_1 = self._select_distances_from_pairs(
                distances1, (pairs2, None))

            distance1_2 = self._sig_error(sig1_1, sig1_2)
            distance2_1 = self._sig_error(sig2_1, sig2_2)
            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1

        # Normalize distance by batch size
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        distance = distance / batch_size

        if self.return_additional_metrics:
            return distance, distance_components
        else:
            return distance


class GromovWassersteinDistance(losses.Loss):
    """Gromov-Wasserstein distance with POT"""
    def __init__(
        self,
        method='gw',
        eps=1e-8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dist_func = OT_DIST.get(method)
        self.eps = eps

    def _compute_distance_matrix(self, x):
        x_flat = tf.reshape(x, [tf.shape(x)[0], -1])
        # Custom l2_norm because tf.norm can cause problems with gradients
        distances = l2_norm(x_flat[:, None] - x_flat, axis=2, eps=self.eps)
        return distances

    def call(self, y_true, y_pred):
        """Return optimal transport distance of two data spaces.

        Args:
            y_true: Coordinates in space 1 (x)
            y_pred: Coordinates in space 2 (latent)

        Returns:
            distance
        """
        distances1 = self._compute_distance_matrix(y_true)
        distances1 = distances1 / tf.math.reduce_max(distances1)
        distances2 = self._compute_distance_matrix(y_pred)
        distances2 = distances2 / tf.math.reduce_max(distances2)
        return self.dist_func(distances1, distances2)


LOSSES = {
    'negative_binomial': NegativeBinomial,
    'zinb': ZINB,
    'topological': TopologicalSignatureDistance(),
    'mmd': MaximumMeanDiscrepancy()
}


def get(identifier: Union[Callable, str]) -> Callable:
    """Returns loss function
    Arguments:
        identifier: Function or string
    Returns:
        Function corresponding to the input string or input function.
    """
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif identifier in LOSSES.keys():
        return LOSSES.get(identifier)
    else:
        return losses.get(identifier)
