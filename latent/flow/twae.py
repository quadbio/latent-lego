'''Tensorflow Twin (Variational) Autoencoder Models'''

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
import tensorflow.keras.layers as layers
from tensorflow.keras.losses import MeanSquaredError, Poisson

from .layers import DenseBlock, MMDCritic
from .losses import MaximumMeanDiscrepancy
from .encoder import VariationalEncoder
from .ae import Autoencoder, PoissonAutoencoder
from .ae import NegativeBinomialAutoencoder, ZINBAutoencoder


class TwinAutoencoder(Model):
    '''Twin autoencoder that joins two autoencoders in a shared latent space'''
    def __init__(
        self,
        models,
        kernel_method = 'multiscale_rbf',
        mmd_weight = 1.0,
        **kwargs
    ):
        super().__init__()
        self.mmd_weight = mmd_weight
        self.kernel_method = kernel_method

        # Define components
        self.ae1, self.ae2 = models
        self.critic_layer = MMDCritic(
            self.ae1.latent_dim,
            weight = self.mmd_weight,
            n_conditions = 2,
            kernel_method = self.kernel_method,
            **kwargs
        )

    def encode(self, inputs):
        in1, in2 = inputs
        x1, sf1 = in1
        x2, sf2 = in2
        latent1 = self.ae1.encoder(x1)
        latent2 = self.ae2.encoder(x2)


    def critic(self, latent1, latent2, split_output=True):
        # Join latent spaces and assign labels
        shared_latent = layers.concatenate([latent1, latent2], axis=0)
        labels = tf.concat(
            [tf.zeros(tf.shape(latent1)[0]), tf.ones(tf.shape(latent2)[0])],
            axis = 0
        )
        labels = tf.cast(labels, tf.int32)
        # Apply critic
        shared_latent = self.critic_layer([shared_latent, labels])
        if split_output:
            # Split latent space again
            latent1, latent2 = tf.dynamic_partition(shared_latent, labels, 2)
            return latent1, latent2
        else:
            return shared_latent, labels

    def call(self, inputs):
        in1, in2 = inputs
        x1, sf1 = in1
        x2, sf2 = in2
        # Map to latent
        latent1 = self.ae1.encoder(x1)
        latent2 = self.ae2.encoder(x2)
        # Critic joins, adds loss, and splits
        latent1, latent2 = self.critic(latent1, latent2)
        # Reconstruction loss should be added by the decoders
        out1 = self.ae1.decoder(x1, latent1, sf1)
        out2 = self.ae2.decoder(x2, latent2, sf2)
        return out1, out2

    def transform(self, inputs, split_output=False):
        x1, x2 = inputs
        # Map to latent
        latent1 = self.ae1.encoder(x1)
        latent2 = self.ae2.encoder(x2)
        # Critic joins, adds loss, and splits
        latent1, latent2 = self.critic(latent1, latent2, split_output=split_output)
        return latent1, latent1

    def compile(self, optimizer='adam', loss=None, **kwargs):
        '''Compile model with default loss and omptimizer'''
        return super().compile(loss=loss, optimizer=optimizer, **kwargs)
