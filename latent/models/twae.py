"""Tensorflow Twin (Variational) Autoencoder Models"""

import tensorflow as tf
import tensorflow.keras as keras
from typing import Union, Tuple

from latent.layers import MMDCritic, CRITICS


class TwinAutoencoder(keras.Model):
    """Twin autoencoder that joins two autoencoders in a shared latent space"""
    def __init__(
        self,
        models: Tuple[keras.Model, keras.Model],
        critic: Union[str, keras.Model] = 'mmd',
        critic_weight: float = 1.,
        n_conditions: int = None,
        **kwargs
    ):
        super().__init__()
        self.critic_weight = tf.Variable(critic_weight, trainable=False)
        self.n_conditions = n_conditions

        # Define components
        self.ae1, self.ae2 = models
        # Change rec loss name if the same
        if self.ae1.decoder.loss_name == self.ae2.decoder.loss_name:
            self.ae1.decoder.loss_name = f'{self.ae1.decoder.loss_name}_1'
            self.ae2.decoder.loss_name = f'{self.ae2.decoder.loss_name}_2'

        has_cond_components = self.ae1._use_conditions() and self.ae2._use_conditions()
        self.use_conditions = self.n_conditions is not None
        # Set condition usage in components is not already set
        if self.use_conditions and not has_cond_components:
            rna_ae.use_conditions = True
            atac_ae.use_conditions = True

        if isinstance(critic, str):
            critic = CRITICS.get(critic, MMDCritic)

        self.critic_layer = critic(
            weight=self.critic_weight,
            n_conditions=2,
            n_groups=self.n_conditions,
            **kwargs
        )

    def critic(self, inputs, latents, split_output=True):
        # Join latent spaces and assign labels
        shared_latent = tf.concat(latents, axis=0)
        labels = tf.concat(
            [tf.zeros(tf.shape(latents[0])[0]), tf.ones(tf.shape(latents[1])[0])],
            axis=0
        )
        labels = tf.cast(labels, tf.int32)
        # If conditions are given, we apply the MMD loss for each separately
        if self.use_conditions:
            # Format condition labels
            cond_labels = [tf.squeeze(tf.argmax(i['cond'], axis=-1)) for i in inputs]
            cond_labels = tf.cast(tf.concat(cond_labels, axis=0), tf.int32)
            # Apply critic for each group
            shared_latent = self.critic_layer([shared_latent, labels, cond_labels])
        else:
            # Apply critic
            shared_latent = self.critic_layer([shared_latent, labels])
        # Split latent space again
        latent1, latent2 = tf.dynamic_partition(shared_latent, labels, 2)
        return latent1, latent2

    def call(self, inputs):
        in1, in2 = inputs
        # Unpack inputs for each autoencoder
        in1 = self.ae1.unpack_inputs(in1)
        in2 = self.ae2.unpack_inputs(in2)
        # Map to latent
        latent1 = self.ae1.encode(in1)
        latent2 = self.ae2.encode(in2)
        # Critic joins, adds loss, and splits
        latent1, latent2 = self.critic([in1, in2], [latent1, latent2])
        # Reconstruction loss should be added by the decoders
        out1 = self.ae1.decode(in1, latent1)
        out2 = self.ae2.decode(in2, latent2)
        return out1, out2

    def transform(self, inputs, join_output=True):
        x1, x2 = inputs
        # Map to latent
        latent1 = self.ae1.encoder(x1)
        latent2 = self.ae2.encoder(x2)
        outputs = [latent1, latent2]
        if join_output:
            outputs = tf.concat(outputs, axis=0)
            labels = tf.concat(
                [tf.zeros(tf.shape(latent1)[0]), tf.ones(tf.shape(latent2)[0])],
                axis=0
            )
            labels = tf.cast(labels, tf.int32)
            return outputs.numpy(), labels.numpy()
        else:
            return outputs.numpy()

    def compile(self, optimizer='adam', loss=None, **kwargs):
        """Compile model with default loss and optimizer"""
        return super().compile(loss=loss, optimizer=optimizer, **kwargs)
