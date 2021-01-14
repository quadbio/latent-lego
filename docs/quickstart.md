# Quickstart

LatentLego is like a lego box with building blocks for autoencoders. These building blocks are hierarchically assembled into larger components, each of which can be used in a modular way to define a model. To make this more clear, lets start with some examples.

## Using pre-defined models

LatentLego hosts a model zoo with a number of common autoencoder architectures ready to use. These are the largest components in LatentLego and therefore on top of the hierarchy. They are subclassed Keras `Model` objects and understand common methods like `.fit()` and `.predict()`. So the quickest way to train an autoencoder with LatentLego is simply

```python
import numpy as np
from latent.models import Autoencoder

x_train = np.array([np.random.poisson(lam=5, size=100) for i in range(100)])

# Creates an autoencoder model with a 2-dimensional latent space
ae = Autoencoder(latent_dim=2, x_dim=x_train.shape[1], activation='relu')

# Compiled the model with a poisson loss and a Adam optimizer
ae.compile(loss='mse', optimizer='adam')

# Trains the model on x_train
ae.fit(x_train, epochs=10, batch_size=10)
```

This trained autoencoder model can now be used to yield a low dimensional representation of the input (or new) data. In line with the [scikit-learn](https://scikit-learn.org/) API, autoencoder models in LatentLego have a `.transform()` method, which returns the low dimensional representation.

```python
x_latent = ae.transform(x_train)
```

Using `.predict()` gives us the reconstructred input data after being passed through the bottleneck. This representation can be used for denoising of single-cell data ([Erslan 2019](https://www.nature.com/articles/s41467-018-07931-2)).

```python
x_recon = ae.predict(x_train)
```

## Defining models by combining encoder and decoder

The second way to define a model with LatentLego is to pick any of the provided `encoder` and `decoder` models and combine them using the `Autoencoder` data structure. For instance, if you want to create a variational autoencoder with a negative binomial reconstruction loss you can simply combine a `VariationalEncoder` with a `NegativeBinomialDecoder`:

```python
from latent.modules import VariationalEncoder, NegativeBinomialDecoder

# Creates a VariationalEncoder with a standard normal prior
encoder = VariationalEncoder(latent_dim=20, prior='normal', kld_weight=0.01)
# Creates a NegativeBinomialDecoder with a constant dispersion estimate
decoder = NegativeBinomialDecoder(x_dim=x_train.shape[1], dispersion='constant')

# Constructs an Autoencoder object with predefined encoder and decoder
ae = Autoencoder(
	encoder=encoder,
	decoder=decoder
)
ae.compile()

# Fit input data using size factors for each cell
x_sf = np.array([1.]*100)
ae.fit([x_train, x_sf], epochs=10, batch_size=10)
```

You might have noticed that this time we did not define a loss function. This is because some encoder and decoder models take care of the model loss on their own. Generally in LatentLego, encoders are responsible for adding losses related to the latent space, such as the KLDivergence loss of a VAE and decoders add the reconstruction loss. If you anyway decide to pass a loss function to `.compile()` it will be added to the reconstruction loss defined by the decoder.

The `Autoencoder` data structure subclasses the Keras `Model` and takes care of providing additional methods like `.transform()` and handling cases with multiple inputs like conditions or size factors. But since `encoder` and `decoder` models are just Keras `Model` objects themselves, they can also be used with the Keras Functional API:

```python
from tensorflow.keras import Input, Model
from latent.modules import Encoder, Decoder

x = Input(shape=(x_train.shape[1],))
x_latent = Encoder(latent_dim=2)(x)

# Decoders take the latent and the original data space as the input
# so they can add the reconstruction loss
x_rec = Decoder(x_dim=x_train.shape[1])([x, x_latent])
ae = Model(inputs=x, outputs=x_rec)

ae.compile(loss='mse', optimizer='adam', run_eagerly=False)
ae.fit(x=x_train, y=x_train, epochs=10, batch_size=10)
```


## Building autoencoders from scratch

Lastly, if you want to implement a model that can not be directly assembled in LatentLego, you may want to implement the encoder/decoder or both from scratch. For this, we also provide some convenient lower-level components that expand the reportoire of Keras and TensorFlow. For instance, the `DenseStack` layer strings together a sequence of `DenseBlock` layers and take care of batch normalization, dropout and conditional injection. We can combine it with some [TensorFlow Probability](https://www.tensorflow.org/probability) magic and stack it on top of a `NegativeBinomialDecoder` to obtain a VAE with negative binomial loss.

```python
import tensorflow as tf
from tensorflow.keras import Sequential, Input, Model
import tensorflow_probability as tfp
tfpl = tfp.layers
tfd = tfp.distributions
from latent.layers import DenseStack
from latent.modules import NegativeBinomialDecoder

latent_dim = 2
# Initiate standard normal prior
prior = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_dim), scale=1),
	reinterpreted_batch_ndims=1)

# Construct encoder with sequential model
encoder = Sequential([
	DenseStack(hidden_units=[256,128,56], batchnorm=True, dropout_rate=0.2),
	tf.keras.layers.Dense(
		tfpl.MultivariateNormalTriL.params_size(latent_dim),
		activation='linear'),
	tfpl.MultivariateNormalTriL(
		latent_dim,
		activity_regularizer=tfpl.KLDivergenceRegularizer(prior))
])
# Initiate a NB decoder with a per-gene dispersion estimate
decoder = NegativeBinomialDecoder(x_dim=x_train.shape[1], dispersion='gene')

x_in = Input(shape=(x_train.shape[1],))
sf_in = Input(shape=(1,))
x_latent = encoder(x_in)
x_rec = decoder([x_in, x_latent, sf_in])
nbvae = Model([x_in, sf_in], x_rec)

# We specify no loss because NB loss is handled by the decoder
nbvae.compile(loss=None, optimizer='sgd')
nbvae.fit(x=[x_train, x_sf], y=x_train, epochs=10, batch_size=10)
```
