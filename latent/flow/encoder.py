'''Tensorflow implementations of encoder models'''

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1_l2
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses

import tensorflow_probability as tfp
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors

from .activations import clipped_softplus, clipped_exp
from .layers import ColwiseMult, DenseStack, PseudoInputs, Sampling, DISTRIBUTIONS


class Encoder(keras.Model):
    '''Classical encoder model'''
    def __init__(
        self,
        latent_dim = 50,
        name='encoder',
        dropout_rate = 0.1,
        batchnorm = True,
        l1 = 0.,
        l2 = 0.,
        hidden_units = [128, 128],
        activation = 'leaky_relu',
        initializer = 'glorot_normal',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.batchnorm =  batchnorm
        self.l1 = l1
        self.l2 = l2
        self.activation = activation
        self.hidden_units =  hidden_units
        self.initializer = keras.initializers.get(initializer)

        # Define components
        self.dense_stack = DenseStack(
            name = self.name,
            dropout_rate = self.dropout_rate,
            batchnorm = self.batchnorm,
            l1 = self.l1,
            l2 = self.l2,
            activation = self.activation,
            initializer = self.initializer,
            hidden_units = self.hidden_units
        )
        self.final_layer = layers.Dense(
            self.latent_dim, name = 'encoder_final',
            kernel_initializer = self.initializer
        )
        self.final_act = layers.Activation('linear', name='encoder_final_activation')

    def call(self, inputs):
        '''Full forward pass through model'''
        h = self.dense_stack(inputs)
        h = self.final_layer(h)
        outputs = self.final_act(h)
        return outputs


class VariationalEncoder(Encoder):
    '''Variational encoder'''
    def __init__(
        self,
        kld_weight = 1e-5,
        prior = 'normal',
        latent_dist = 'independent_normal',
        iaf_units = [256, 256],
        vamp_pseudoinputs = 200,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.kld_weight = tf.Variable(kld_weight, trainable=False)
        self.prior = prior
        self.iaf_units = iaf_units
        self.vamp_pseudoinputs = vamp_pseudoinputs
        self.latent_dist = latent_dist

        # Define components
        self.latent_dist_layer = DISTRIBUTIONS.get(latent_dist, tfpl.IndependentNormal)
        self.dist_param_layer = layers.Dense(
            self.latent_dist_layer.params_size(self.latent_dim),
            name = 'encoder_dist_params',
            kernel_initializer = self.initializer
        )
        self.sampling = self.latent_dist_layer(self.latent_dim)

        # Independent() reinterprets each latent_dim as an independent distribution
        unit_mvn = tfd.Independent(
            tfd.Normal(loc=tf.zeros(self.latent_dim), scale=1.),
            reinterpreted_batch_ndims = 1
        )

        if self.prior == 'normal':
            self.prior_dist = unit_mvn

        elif self.prior == 'iaf':
            # Inverse autoregressive flow (Kingma et al. 2016)
            made = tfb.AutoregressiveNetwork(params=2, hidden_units=self.iaf_units)
            self.prior_dist = tfd.TransformedDistribution(
                distribution = unit_mvn,
                bijector = tfb.Invert(tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=made))
            )

        elif self.prior == 'vamp':
            # Variational mixture of posteriors (VAMP) prior (Tomczak & Welling 2018)
            self.pseudo_inputs = PseudoInputs(n_inputs=self.vamp_pseudoinputs)

        elif self.prior == 'vmf':
            # Hyperspherical von Mises-Fisher prior (Davidson et al. 2018)
            pass

    def call(self, inputs):
        '''Full forward pass through model'''
        h = self.dense_stack(inputs)
        dist_params = self.dist_param_layer(h)
        outputs = self.sampling(dist_params)
        if self.prior == 'vamp':
            # VAMP prior depends on input, do we have to add it here
            prior_dist = self._vamp_prior(inputs)
            # Original implementation performs additional logsumexp on prior log_prob
            # this would require to compute KLDivergence manually, e.g.:
            # z = outputs.sample()
            # tf.math.reduce_mean(outputs.log_prob(z))
            # - tf.math.reduce_logsumexp(prior.log_prob(z)))
            # This would also solve the problem of dimension mismatch
        else:
            prior_dist = self.prior_dist
        kld_regularizer = tfpl.KLDivergenceRegularizer(
            prior_dist,
            weight = self.kld_weight,
            test_points_reduce_axis = None
        )
        kld_loss = kld_regularizer(outputs)
        # Add losses manually to better monitor them
        self.add_loss(kld_loss)
        self.add_metric(kld_loss, name='kld_loss')
        return outputs

    def _vamp_prior(self, inputs):
        # Inputs are needed to infer shape
        # and to ensure a connected graph
        h = self.pseudo_inputs(inputs)
        h = self.dense_stack(h)
        dist_params = self.dist_param_layer(h)
        outputs = self.sampling(dist_params)
        return outputs
