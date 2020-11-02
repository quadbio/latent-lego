'''Tensorflow Twin (Variational) Autoencoder Models'''

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
import tensorflow.keras.layers as layers
from tensorflow.keras.losses import MeanSquaredError, Poisson

from .core import DenseBlock
from .losses import MaximumMeanDiscrepancy
from .encoder import VariationalEncoder
from .ae import Autoencoder, PoissonAutoencoder
from .ae import NegativeBinomialAutoencoder, ZINBAutoencoder

class TwinAutoencoder(Model):
    '''Twin autoencoder that joins two autoencoders in a shared latent space'''
    def __init__(self, models, losses, **kwargs):
        super().__init__(**kwargs)

        # Define components
        self.ae1, self.ae2 = models
        self.loss1, self.loss2 = losses

    def call(self, inputs):
        in1, in2 = inputs
        latent1 = self.ae1.encoder(in1)
        latent2 = self.ae2.encoder(in2)
        # Concatenate along samples
        shared_latent = layers.concatenate([latent1, latent2], axis=0)
        labels = K.concatenate(
            K.zeros(K.shape(latent1)[0]),
            K.ones(K.shape(latent2)[0])
        )
        # Here we split again right away, but actually we need some loss that
        # enforces a joint latent space
        latent1, latent2 = tf.dynamic_partition(shared_latent, labels, 2)
        out1 = self.ae1.decoder(latent1)
        out2 = self.ae2.decoder(latent2)
        # Losses have to be added here because they are separate for each input
        self.add_loss(self.loss1(in1, out1))
        self.add_loss(self.loss2(in2, out2))
        return out1, out2

    def transform(self, inputs):
        '''Map data (x) to shared latent space (z)'''
        in1, in2 = inputs
        latent1 = self.ae1.encoder.predict(in1)
        latent2 = self.ae2.encoder.predict(in2)
        # Concatenate along samples
        shared_latent = layers.concatenate([latent1, latent2], axis=0)
        return shared_latent

    def compile(self, optimizer='adam', loss=None, **kwargs):
        '''Compile model with default loss and omptimizer'''
        return super().compile(loss=loss, optimizer=optimizer, **kwargs)


class MMDTwinAutoencoder(TwinAutoencoder):
        '''Twin autoencoder that joins two autoencoders in a shared latent space'''
        def __init__(
            self,
            kernel_method = 'multiscale_rbf',
            mmd_weight = 1,
            **kwargs
        ):
            super().__init__(**kwargs)
            self.mmd_weight = mmd_weight

            # Define components
            # Layer after which MMD loss is applied
            # -> Joins latent spaces
            self.mmd_layer = DenseBlock(
                20,
                dropout_rate = 0
            )
            self.mmd_loss = MaximumMeanDiscrepancy(
                n_conditions = 2,
                weight = self.mmd_weight
            )

        def call(self, inputs):
            in1, in2 = inputs
            latent1 = self.ae1.encoder(in1)
            latent2 = self.ae2.encoder(in2)
            # Concatenate along samples
            shared_latent = layers.concatenate([latent1, latent2], axis=0)
            labels = tf.concat(
                [tf.zeros(tf.shape(latent1)[0]), tf.ones(tf.shape(latent2)[0])],
                axis = 0
            )
            labels = tf.cast(labels, tf.int32)
            mmd_latent = self.mmd_layer(shared_latent)
            # MMD loss to enforce shared latent space after MMD layer
            mmd_loss = self.mmd_loss((None, labels), mmd_latent)
            self.add_loss(mmd_loss)
            self.add_metric(mmd_loss, name='mmd_loss')
            latent1, latent2 = tf.dynamic_partition(mmd_latent, labels, 2)
            out1 = self.ae1.decoder(latent1)
            out2 = self.ae2.decoder(latent2)
            # Losses have to be added here because they are separate for each input
            recon_loss_1 = tf.reduce_mean(self.loss1(in1, out1))
            recon_loss_2 = tf.reduce_mean(self.loss1(in2, out2))
            self.add_loss(recon_loss_1)
            self.add_metric(recon_loss_1, name='recon_loss_1')
            self.add_loss(recon_loss_2)
            self.add_metric(recon_loss_2, name='recon_loss_2')
            return out1, out2
