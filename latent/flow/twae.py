'''Tensorflow Twin (Variational) Autoencoder Models'''

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
import tensorflow.keras.layers as layers
from tensorflow.keras.losses import MeanSquaredError, Poisson

from .layers import DenseBlock, MMDCritic, CRITICS
from .losses import MaximumMeanDiscrepancy
from .encoder import VariationalEncoder
from .ae import Autoencoder, PoissonAutoencoder
from .ae import NegativeBinomialAutoencoder, ZINBAutoencoder


class TwinAutoencoder(keras.Model):
    '''Twin autoencoder that joins two autoencoders in a shared latent space'''
    def __init__(
        self,
        models,
        critic = 'mmd',
        kernel_method = 'multiscale_rbf',
        critic_weight = 1.0,
        critic_units = None,
        **kwargs
    ):
        super().__init__()
        self.critic_weight = critic_weight
        self.kernel_method = kernel_method
        self.critic_units = critic_units

        # Define components
        self.ae1, self.ae2 = models

        if isinstance(critic, str):
            critic = CRITICS.get(critic, MMDCritic)

        self.critic_layer = critic(
            self.ae1.latent_dim,
            weight = self.critic_weight,
            kernel_method = self.kernel_method,
            critic_units = self.critic_units,
            **kwargs
        )

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
        latent1 = self.ae1.encode(x1)
        latent2 = self.ae2.encode(x2)
        # Critic joins, adds loss, and splits
        latent1, latent2 = self.critic(latent1, latent2)
        # Reconstruction loss should be added by the decoders
        out1 = self.ae1.decode(x1, latent1, sf1, loss_name='rec_loss_1')
        out2 = self.ae2.decode(x2, latent2, sf2, loss_name='rec_loss_2')
        return out1, out2

    def transform(self, inputs, split_output=False):
        x1, x2 = inputs
        # Map to latent
        latent1 = self.ae1.encoder(x1)
        latent2 = self.ae2.encoder(x2)
        # Critic joins, adds loss, and splits
        outputs = self.critic(latent1, latent2, split_output=split_output)
        return [out.numpy() for out in outputs]

    def compile(self, optimizer='adam', loss=None, **kwargs):
        '''Compile model with default loss and optimizer'''
        return super().compile(loss=loss, optimizer=optimizer, **kwargs)
