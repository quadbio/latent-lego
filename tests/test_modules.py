import pytest
import tensorflow.keras as keras
from latent.models import Autoencoder
from latent.modules import *

nx = 600
nd = 200
X = np.random.uniform(low=0, high=30, size=(nx, nd)).astype(np.float32)
sf = np.ones(nx)
cond = np.random.randint(3, size=nx).astype(np.float32)
ld = 20

enc = Encoder(latent_dim=ld, reconstruction_loss='mse')
c_enc = Encoder(latent_dim=ld, conditional='all')
var_enc = VariationalEncoder(latent_dim=ld)

dec = Decoder(x_dim=X.shape[1])
nb_dec = NegativeBinomialDecoder(x_dim=X.shape[1])
c_dec = Decoder(x_dim=X.shape[1], conditional='first')


def test_autoencoder():
    ae = Autoencoder(enc, dec)
    ae.compile()
    ae.fit(X, batch_size=50, epochs=1)
    lat = ae.transform(X)
    assert X.shape[0] == lat.shape[0]
    assert lat.shape[1] == ld
    rec = ae.predict(X)
    assert rec.shape == X.shape


def test_variational_autoencoder():
    ae = Autoencoder(var_enc, dec)
    ae.compile()
    ae.fit(X, batch_size=50, epochs=1)
    lat = ae.transform(X)
    assert X.shape[0] == lat.shape[0]
    assert lat.shape[1] == ld
    rec = ae.predict(X)
    assert rec.shape == X.shape


def test_conditional_autoencoder():
    ae = Autoencoder(enc, c_dec)
    ae.compile()
    assert ae._use_conditions() == True
    assert ae._conditional_decoder() == True
    assert ae._conditional_encoder() == False
    ae.fit([X, cond], batch_size=50, epochs=1)
    lat = ae.transform([X, cond])
    assert X.shape[0] == lat.shape[0]
    assert lat.shape[1] == ld
    rec = ae.predict([X, cond])
    assert rec.shape == X.shape

    ae = Autoencoder(c_enc, dec)
    ae.compile()
    assert ae._use_conditions() == True
    assert ae._conditional_decoder() == False
    assert ae._conditional_encoder() == True
    ae.fit([X, cond], batch_size=50, epochs=1)
    lat = ae.transform([X, cond])
    assert X.shape[0] == lat.shape[0]
    assert lat.shape[1] == ld
    rec = ae.predict([X, cond])
    assert rec.shape == X.shape

    ae = Autoencoder(c_enc, c_dec)
    ae.compile()
    assert ae._use_conditions() == True
    assert ae._conditional_decoder() == True
    assert ae._conditional_encoder() == True
    ae.fit([X, cond], batch_size=50, epochs=1)
    lat = ae.transform([X, cond])
    assert X.shape[0] == lat.shape[0]
    assert lat.shape[1] == ld
    rec = ae.predict([X, cond])
    assert rec.shape == X.shape
