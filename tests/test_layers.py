import pytest
import numpy as np
import tensorflow.keras as keras
from latent.layers import *

nx = 600
nd = 200
X = np.random.uniform(low=0, high=30, size=(nx, nd)).astype(np.float32)
cond = np.random.randint(3, size=nx).astype(np.float32)
ld = 20


def test_colwise_mult():
    x = np.array([[1, 2, 3], [1, 2, 3], [3, 2, 1]])
    y = np.array([1, 2, 3])
    exp_res = np.array([[1, 2, 3], [2, 4, 6], [9, 6, 3]])
    rmult = RowwiseMult()
    res = rmult([x, y]).numpy()
    assert np.allclose(res, exp_res)


def test_shared_dispersion():
    units = 20
    sdisp = SharedDispersion(units)
    res = sdisp(X).numpy()
    assert res.shape == (nx, units)
    assert np.all(np.all(res == res[0, :], axis=0))


def test_constant():
    units = 2
    x = np.array([[1, 2, 3], [1, 2, 3], [3, 2, 1]])
    exp_res = np.array([[2, 2], [2, 2], [2, 2]])
    const = Constant(units, constant=2, activation="linear")
    res = const(x).numpy()
    assert np.allclose(res, exp_res)


def test_sequential_api():
    enc = keras.Sequential(
        [
            DenseBlock(units=512, activation="relu"),
            DenseStack(hidden_units=[256, 128, 56], batchnorm=True, dropout_rate=0.2),
            DenseBlock(units=20, activation="linear"),
        ]
    )
    dec = keras.Sequential(
        [
            DenseBlock(units=20, activation="relu"),
            DenseStack(hidden_units=[56, 128, 256], batchnorm=True, dropout_rate=0.1),
            DenseBlock(units=200, activation="linear"),
        ]
    )
    x_in = keras.Input(shape=(X.shape[1],))
    x_latent = enc(x_in)
    x_rec = dec(x_latent)
    ae = keras.Model(x_in, x_rec)
    ae.compile(loss="mse", optimizer="sgd")
    ae.fit(x=X, y=X, epochs=10, batch_size=10)
