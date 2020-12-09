import inspect
import warnings
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
kernels = tfp.math.psd_kernels

# Only needed for GW-OT
try:
    import ot
except ModuleNotFoundError:
    warnings.warn('POT package not available.')


# Kernels
# Multi-scale RBF kernel modified from https://github.com/theislab/scarches
def ms_rbf_kernel(x, y):
    """Multi-scale RBF kernel"""
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
        1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    dist = squared_distance(x, y)
    s = tf.tensordot(beta, tf.reshape(dist, (1, -1)), axes=0)
    return tf.reshape(tf.math.reduce_sum(tf.exp(-s), 0), tf.shape(dist)) / len(sigmas)


def rbf_kernel(x, y):
    """Radial Basis Function or Exponentiated Quadratic kernel"""
    kernel = kernels.ExponentiatedQuadratic()
    return kernel.matrix(x, y)


def rq_kernel(x, y):
    """Rational Quadratic kernel"""
    kernel = kernels.RationalQuadratic()
    return kernel.matrix(x, y)


KERNELS = {
    'ms_rbf': ms_rbf_kernel,
    'rbf': rbf_kernel,
    'rq': rq_kernel
}


# Methods for calculating lower-dimensional persistent homology.
# Implementations adapted from https://github.com/BorgwardtLab/topological-autoencoders
class UnionFind:
    """
    An implementation of a UnionFind class. The class performs path
    compression by default. It uses integers for storing one disjoint
    set, assuming that vertices are zero-indexed.
    """

    def __init__(self, n_vertices):
        """
        Initializes an empty Union--Find data structure for a given
        number of vertices.
        """

        self._parent = np.arange(n_vertices, dtype=int)

    def find(self, u):
        """
        Finds and returns the parent of u with respect to the hierarchy.
        """

        if self._parent[u] == u:
            return u
        else:
            # Perform path collapse operation
            self._parent[u] = self.find(self._parent[u])
            return self._parent[u]

    def merge(self, u, v):
        """
        Merges vertex u into the component of vertex v. Note the
        asymmetry of this operation.
        """

        if u != v:
            self._parent[self.find(u)] = self.find(v)

    def roots(self):
        """
        Generator expression for returning roots, i.e. components that
        are their own parents.
        """

        for vertex, parent in enumerate(self._parent):
            if vertex == parent:
                yield vertex


def _persistent_homology(matrix):
    """Performs persistent homology calculation"""
    n_vertices = matrix.shape[0]
    uf = UnionFind(n_vertices)

    triu_indices = np.triu_indices_from(matrix)
    edge_weights = matrix[triu_indices]
    edge_indices = np.argsort(edge_weights, kind='stable')

    # 1st dimension: 'source' vertex index of edge
    # 2nd dimension: 'target' vertex index of edge
    persistence_pairs = []

    for edge_index, edge_weight in zip(edge_indices, edge_weights[edge_indices]):

        u = triu_indices[0][edge_index]
        v = triu_indices[1][edge_index]

        younger_component = uf.find(u)
        older_component = uf.find(v)

        # Not an edge of the MST, so skip it
        if younger_component == older_component:
            continue
        elif younger_component > older_component:
            uf.merge(v, u)
        else:
            uf.merge(u, v)

        if u < v:
            persistence_pairs.append((u, v))
        else:
            persistence_pairs.append((v, u))

    # Return empty cycles component
    # -> Changed to not return cycles
    return np.array(persistence_pairs, dtype=np.int64)


def persistent_homology(matrix):
    return tf.numpy_function(
        _persistent_homology, [matrix], tf.int64)


# Methods for Gromov-Wasserstein distance calculations
def _gromov_wasserstein_distance(x, y):
    x_p = ot.unif(x.shape[0])
    y_q = ot.unif(y.shape[0])
    gw_dist = ot.gromov.gromov_wasserstein2(
        x, y, x_p, y_q, loss_fun='kl_loss')
    return np.array(gw_dist, dtype=np.float32)


def _entropic_gromov_wasserstein_distance(x, y):
    x_p = ot.unif(x.shape[0])
    y_q = ot.unif(y.shape[0])
    gw_dist = ot.gromov.entropic_gromov_wasserstein2(
        x, y, x_p, y_q, loss_fun='kl_loss')
    return np.array(gw_dist, dtype=np.float32)


def gromov_wasserstein_distance(x, y):
    return tf.numpy_function(
        _gromov_wasserstein_distance, [x, y], tf.float32)


def entropic_gromov_wasserstein_distance(x, y):
    return tf.numpy_function(
        _entropic_gromov_wasserstein_distance, [x, y], tf.float32)


OT_DIST = {
    'gw': gromov_wasserstein_distance,
    'entropic_gw': entropic_gromov_wasserstein_distance
}


# Other
def squared_distance(x, y):
    r = tf.expand_dims(x, axis=1)
    return tf.math.reduce_sum(tf.math.square(r - y), axis=-1)


def nan2zero(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)


def nan2inf(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x) + np.inf, x)


def clip_nonzero(x, min_val=1e-8):
    clipped_x = tf.clip_by_value(x, min_val, tf.math.reduce_max(x))
    return tf.where(x > 0, clipped_x, x)


def nelem(x):
    nelem = tf.math.reduce_sum(tf.cast(~tf.math.is_nan(x), tf.float32))
    return tf.cast(tf.where(tf.math.equal(nelem, 0.), 1., nelem), x.dtype)


def slice_matrix(matrix, row_idx, col_idx):
    row_select = tf.gather(matrix, row_idx, axis=0)
    col_select = tf.gather(row_select, col_idx, axis=-1)
    return col_select


def l2_norm(x, axis=2, eps=1e-8):
    return tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis) + eps)


def size_factors(x):
    n = x.sum(1)
    return n / np.median(n)


def delegates(to=None, keep=False):
    """Decorator: replace `**kwargs` in signature with params from `to`"""
    def _f(f):
        if to is None:
            to_f, from_f = f.__base__.__init__, f.__init__
        else:
            to_f, from_f = to, f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop('kwargs')
        s2 = {k:v for k,v in inspect.signature(to_f).parameters.items()
            if v.default != inspect.Parameter.empty and k not in sigd}
        sigd.update(s2)
        if keep:
            sigd['kwargs'] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f
    return _f
