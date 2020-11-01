'''Tensorflow Twin (Variational) Autoencoder Models'''

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.losses import MeanSquaredError, Poisson

from .ae import Autoencoder, PoissonAutoencoder
from .ae import NegativeBinomialAutoencoder, ZINBAutoencoder
from .encoder import VariationalEncoder

class TwinAutoencoder(Model):
    def __init__(self, models, **kwargs):
        super().__init__(**kwargs)

        # Define components
        self.ae1 = models[0]
        self.ae2 = models[2]

    def call(self, inputs):
        in1, in2 = inputs
        latent1 = self.ae1.encoder(in1)
        latent2 = self.ae2.encoder(in2)
