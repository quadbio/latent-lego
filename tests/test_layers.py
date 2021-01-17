import pytest
import numpy as np
import tensorflow.keras as keras
from latent.layers import *

nx = 600
nd = 200
X = np.random.uniform(low=0, high=30, size=(nx, nd)).astype(np.float32)
sf = np.ones(nx)
cond = np.random.randint(3, size=nx).astype(np.float32)
ld = 20


def test_colwise_mult():
	x = np.array([[1,2,3], [1,2,3], [3,2,1]])
	y = np.array([1,2,3])
	exp_res = np.array([[1,2,3], [2,4,6], [9,6,3]])
	rmult = RowwiseMult()
	res = rmult([x,y]).numpy()
	assert np.allclose(res, exp_res)


def test_shared_dispersion():
	units = 20
	sdisp = SharedDispersion(units)
	res = sdisp(X).numpy()
	assert res.shape == (nx, units)


def test_model_construction():
	enc = keras.Sequential([
		DenseStack(hidden_units=[256,128,56], batchnorm=True, dropout_rate=0.2),
		DenseBlock(units=20, activation='linear')
	])
	dec = keras.Sequential([
		DenseStack(hidden_units=[56,128,256], batchnorm=True, dropout_rate=0.1),
		DenseBlock(units=200, activation='linear')
	])
	x_in = keras.Input(shape=(x_train.shape[1],))
	sf_in = keras.Input(shape=(1,))
	x_latent = enc(x_in)
	x_rec = decoder([x_in, x_latent, sf_in])
	ae = keras.Model([x_in, sf_in], x_rec)
	ae.compile(loss=None, optimizer='sgd')
	ae.fit(x=[x_train, x_sf], y=x_train, epochs=10, batch_size=10)
