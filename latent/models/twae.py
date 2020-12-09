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
        join_conditions: bool = False,
        condition_weight: float = 1.,
        n_conditions: int = 2,
        **kwargs
    ):
        super().__init__()
        self.critic_weight = tf.Variable(critic_weight, trainable=False)
        self.condition_weight = tf.Variable(condition_weight, trainable=False)
        self.n_conditions = n_conditions

        # Define components
        self.ae1, self.ae2 = models
        # Change rec loss name if the same
        if self.ae1.decoder.loss_name == self.ae2.decoder.loss_name:
            self.ae1.decoder.loss_name = f'{self.ae1.decoder.loss_name}_1'
            self.ae2.decoder.loss_name = f'{self.ae2.decoder.loss_name}_2'

        self.join_conditions = (join_conditions and self.ae1._use_conditions()
                                and self.ae2._use_conditions())

        if isinstance(critic, str):
            critic = CRITICS.get(critic, MMDCritic)

        self.critic_layer = critic(
            weight=self.critic_weight,
            hidden_units=None,
            n_conditions=2,
            **kwargs
        )
        if self.join_conditions:
            self.condition_layer = critic(
                weight=self.condition_weight,
                hidden_units=None,
                name='condition_critic',
                n_conditions=self.n_conditions,
                **kwargs
            )

    def critic(self, latents, inputs=None, split_output=True):
        # Join latent spaces and assign labels
        shared_latent = tf.concat(latents, axis=0)
        labels = tf.concat(
            [tf.zeros(tf.shape(latents[0])[0]), tf.ones(tf.shape(latents[1])[0])],
            axis=0
        )
        labels = tf.cast(labels, tf.int32)
        # Apply critic
        shared_latent = self.critic_layer([shared_latent, labels])
        # Join conditions
        ###
        # Loss conditional on condition, calcualte loss between modalities only within condition, instead of globally
        ###
        if self.join_conditions and inputs:
            cond_labels = [tf.squeeze(tf.argmax(i['cond'], axis=-1)) for i in inputs]
            cond_labels = tf.concat(cond_labels, axis=0)
            shared_latent = self.condition_layer([shared_latent, cond_labels])
        # Split latent space again
        if split_output:
            latent1, latent2 = tf.dynamic_partition(shared_latent, labels, 2)
            return latent1, latent2
        else:
            return shared_latent, labels

    def call(self, inputs):
        in1, in2 = inputs
        # Unpack inputs for each autoencoder
        in1 = self.ae1.unpack_inputs(in1)
        in2 = self.ae2.unpack_inputs(in2)
        # Map to latent
        latent1 = self.ae1.encode(in1)
        latent2 = self.ae2.encode(in2)
        # Critic joins, adds loss, and splits
        latent1, latent2 = self.critic([latent1, latent2], inputs=[in1, in2])
        # Reconstruction loss should be added by the decoders
        out1 = self.ae1.decode(in1, latent1)
        out2 = self.ae2.decode(in2, latent2)
        return out1, out2

    def transform(self, inputs, split_output=False):
        x1, x2 = inputs
        # Map to latent
        latent1 = self.ae1.encoder(x1)
        latent2 = self.ae2.encoder(x2)
        # Critic joins, adds loss, and splits
        outputs = self.critic([latent1, latent2], split_output=split_output)
        return [out.numpy() for out in outputs]

    def compile(self, optimizer='adam', loss=None, **kwargs):
        """Compile model with default loss and optimizer"""
        return super().compile(loss=loss, optimizer=optimizer, **kwargs)
