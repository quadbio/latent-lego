'''Tensorflow implementations of losses for autoencoders'''

import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras.losses as losses

from .utils import ms_rbf_kernel, rbf_kernel, persistent_homology, slice_matrix
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


# Implementation adapted from https://github.com/BorgwardtLab/topological-autoencoders
class TopologicalSignatureDistance(losses.Loss):
    '''Distance between topological signatures.'''
    def __init__(
        self,
        sort_selected = False,
        use_cycles = False,
        match_edges = None,
        return_additional_metrics = False,
        **kwargs
    ):
        '''Topological signature computation.

        Args:
            p: Order of norm used for distance computation
            use_cycles: Flag to indicate whether cycles should be used
                or not.
        '''
        super().__init__(**kwargs)
        self.use_cycles = use_cycles
        self.match_edges = match_edges
        self.return_additional_metrics = return_additional_metrics

    def _get_pairings(self, distances):
        pairs_0, pairs_1 = persistent_homology(distances)
        return pairs_0, pairs_1

    def _select_distances_from_pairs(self, distance_matrix, pairs):
        # Split 0th order and 1st order features (edges and cycles)
        pairs_0, pairs_1 = pairs
        # slice_matrix utility func because tf does not allow us to slice matrices
        selected_distances = slice_matrix(distance_matrix, pairs_0[:, 0], pairs_0[:, 1])

        if self.use_cycles:
            edges_1 = slice_matrix(distance_matrix, pairs_1[:, 0], pairs_1[:, 1])
            edges_2 = slice_matrix(distance_matrix, pairs_1[:, 2], pairs_1[:, 3])
            edge_differences = edges_2 - edges_1

            selected_distances = tf.concat(
                (selected_distances, edge_differences))

        return selected_distances

    @staticmethod
    def sig_error(signature1, signature2):
        '''Compute distance between two topological signatures.'''
        return tf.reduce_sum((signature1 - signature2)**2, axis=-1)

    @staticmethod
    def _count_matching_pairs(pairs1, pairs2):
        return tf.sets.intersection(pairs1, pairs2)

    @staticmethod
    def _get_nonzero_cycles(pairs):
        all_indices_equal = tf.math.reduce_sum(
            pairs[:, [0]] == pairs[:, 1:], axis=-1) == 3
        return tf.math.reduce_sum(tf.math.logical_not(all_indices_equal))

    @staticmethod
    def _compute_distance_matrix(x, ord=2):
        x_flat = tf.reshape(x, [tf.shape(x)[0], -1])
        distances = tf.norm(x_flat[:, None] - x_flat, axis=2, ord=ord)
        return distances

    def call(self, y_true, y_pred):
        '''Return topological distance of two data spaces.

        Args:
            y_true: Coordinates in space 1 (x)
            y_pred: Coordinates in space 2 (latent)

        Returns:
            distance, [dict(additional outputs)]
        '''
        distances1 = self._compute_distance_matrix(y_true)
        distances1 = distances1 / tf.math.reduce_max(distances1)
        distances2 = self._compute_distance_matrix(y_pred)
        distances2 = distances2 / tf.math.reduce_max(distances2)

        pairs1 = self._get_pairings(distances1)
        pairs2 = self._get_pairings(distances2)

        distance_components = {
            'metrics.matched_pairs_0D': self._count_matching_pairs(
                pairs1[0], pairs2[0])
        }
        # Also count matched cycles if present
        if self.use_cycles:
            distance_components['metrics.matched_pairs_1D'] = \
                self._count_matching_pairs(pairs1[1], pairs2[1])
            nonzero_cycles_1 = self._get_nonzero_cycles(pairs1[1])
            nonzero_cycles_2 = self._get_nonzero_cycles(pairs2[1])
            distance_components['metrics.non_zero_cycles_1'] = nonzero_cycles_1
            distance_components['metrics.non_zero_cycles_2'] = nonzero_cycles_2

        if not self.match_edges:
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            distance = self.sig_error(sig1, sig2)

        elif self.match_edges == 'symmetric':
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            # Selected pairs of 1 on distances of 2 and vice versa
            sig1_2 = self._select_distances_from_pairs(distances2, pairs1)
            sig2_1 = self._select_distances_from_pairs(distances1, pairs2)

            distance1_2 = self.sig_error(sig1, sig1_2)
            distance2_1 = self.sig_error(sig2, sig2_1)

            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1

        elif self.match_edges == 'random':
            # Create random selection in order to verify if what we are seeing
            # is the topological constraint or an implicit latent space prior
            # for compactness
            n_instances = len(pairs1[0])
            pairs1 = tf.concat([
                tf.random.shuffle(n_instances)[:, None],
                tf.random.shuffle(n_instances)[:, None]
            ], axis=1)
            pairs2 = tf.concat([
                tf.random.shuffle(n_instances)[:, None],
                tf.random.shuffle(n_instances)[:, None]
            ], axis=1)

            sig1_1 = self._select_distances_from_pairs(
                distances1, (pairs1, None))
            sig1_2 = self._select_distances_from_pairs(
                distances2, (pairs1, None))

            sig2_2 = self._select_distances_from_pairs(
                distances2, (pairs2, None))
            sig2_1 = self._select_distances_from_pairs(
                distances1, (pairs2, None))

            distance1_2 = self.sig_error(sig1_1, sig1_2)
            distance2_1 = self.sig_error(sig2_1, sig2_2)
            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1

        if self.return_additional_metrics:
            return distance, distance_components
        else:
            return distance
