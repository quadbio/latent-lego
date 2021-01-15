import pytest
from latent.models import *

nx = 600
nd = 200
X = np.random.uniform(low=0, high=30, size=(nx, nd)).astype(np.float32)
sf = np.ones(nx)
cond = np.random.randint(3, size=nx).astype(np.float32)


def check_attributes(model, ce=False, cd=False, c=False, sf=False):
    assert model._conditional_encoder() == ce
    assert model._conditional_decoder() == cd
    assert model._use_conditions() == c
    assert model._use_sf() == sf


def test_autoencoder():
    lat_dim = 18
    ae = Autoencoder(
        x_dim = X.shape[1],
        latent_dim = lat_dim
    )
    ae.compile(optimizer='adam', loss='mse', run_eagerly=False)
    check_attributes(ae)
    ae.fit(
        X,
        batch_size = 50,
        epochs = 2
    )
    lat = ae.transform(X)
    assert X.shape[0] == lat.shape[0]
    assert lat.shape[1] == lat_dim
    rec = ae.predict(X)
    assert rec.shape == X.shape


def test_conditional_autoencoder():
    ae = Autoencoder(
        x_dim = X.shape[1],
        conditional = 'all'
    )
    ae.compile(optimizer='adam', loss='mse', run_eagerly=True)
    check_attributes(ae, ce=True, cd=True, c=True)
    ae.fit(
        [X, cond],
        batch_size = 50,
        epochs = 2
    )
    lat = ae.transform([X, cond])
    assert X.shape[0] == lat.shape[0]
    rec = ae.predict([X, cond])
    assert rec.shape == X.shape

    ae = Autoencoder(
        x_dim = X.shape[1],
        conditional = 'first'
    )
    ae.compile(optimizer='adam', loss='mse', run_eagerly=False)
    check_attributes(ae, ce=True, cd=True, c=True)
    ae.fit(
        [X, cond],
        batch_size = 50,
        epochs = 2
    )
    lat = ae.transform([X, cond])
    assert X.shape[0] == lat.shape[0]
    rec = ae.predict([X, cond])
    assert rec.shape == X.shape


def test_poisson_autoencoder():
    ae = PoissonAutoencoder(
        x_dim = X.shape[1]
    )
    ae.compile(optimizer='adam', loss='mse', run_eagerly=False)
    check_attributes(ae, sf=True)
    ae.fit(
        [X, sf],
        batch_size = 50,
        epochs = 2
    )
    lat = ae.transform(X)
    assert X.shape[0] == lat.shape[0]
    rec = ae.predict([X, sf])
    assert rec.shape == X.shape


def test_nb_autoencoder():
    ae = NegativeBinomialAutoencoder(
        x_dim = X.shape[1],
        dispersion = 'gene'
    )
    ae.compile(optimizer='adam', run_eagerly=False)
    check_attributes(ae, sf=True)
    ae.fit(
        [X, sf],
        batch_size = 50,
        epochs = 2
    )
    lat = ae.transform(X)
    assert X.shape[0] == lat.shape[0]
    rec = ae.predict([X, sf])
    assert rec.shape == X.shape


def test_conditional_zinb_autoencoder():
    ae = NegativeBinomialAutoencoder(
        x_dim = X.shape[1],
        dispersion = 'constant',
        conditional = 'all'
    )
    ae.compile(optimizer='adam', run_eagerly=True)
    check_attributes(ae, ce=True, cd=True, c=True, sf=True)
    ae.fit(
        [X, cond, sf],
        batch_size = 50,
        epochs = 2
    )
    lat = ae.transform([X, cond])
    assert X.shape[0] == lat.shape[0]
    rec = ae.predict([X, cond, sf])
    assert rec.shape == X.shape


def test_topological_autoencoder():
    ae = TopologicalAutoencoder(
        x_dim = X.shape[1]
    )
    ae.compile(optimizer='adam', run_eagerly=True)
    check_attributes(ae)
    ae.fit(
        X,
        batch_size = 50,
        epochs = 2
    )
    lat = ae.transform(X)
    assert X.shape[0] == lat.shape[0]
    rec = ae.predict(X)
    assert rec.shape == X.shape


def test_variational_autoencoder():
    ae = VariationalAutoencoder(
        x_dim = X.shape[1]
    )
    ae.compile(optimizer='adam', run_eagerly=False)
    check_attributes(ae)
    ae.fit(
        X,
        batch_size = 50,
        epochs = 2
    )
    lat = ae.transform(X)
    assert X.shape[0] == lat.shape[0]
    rec = ae.predict(X)
    assert rec.shape == X.shape

    ae = VariationalAutoencoder(
        x_dim = X.shape[1],
        latent_dist = 'multivariate'
    )
    ae.compile(optimizer='adam', run_eagerly=False)
    check_attributes(ae)
    ae.fit(
        X,
        batch_size = 50,
        epochs = 2
    )
    lat = ae.transform(X)
    assert X.shape[0] == lat.shape[0]
    rec = ae.predict(X)
    assert rec.shape == X.shape

    ae = VariationalAutoencoder(
        x_dim = X.shape[1],
        prior = 'iaf',
        iaf_units = [128, 128]
    )
    ae.compile(optimizer='adam', run_eagerly=False)
    check_attributes(ae)
    ae.fit(
        X,
        batch_size = 50,
        epochs = 2
    )
    lat = ae.transform(X)
    assert X.shape[0] == lat.shape[0]
    rec = ae.predict(X)
    assert rec.shape == X.shape

    ae = VariationalAutoencoder(
        x_dim = X.shape[1],
        prior = 'vamp',
        n_pseudoinputs = 30
    )
    ae.compile(optimizer='adam', run_eagerly=False)
    check_attributes(ae)
    ae.fit(
        X,
        batch_size = 50,
        epochs = 2
    )
    lat = ae.transform(X)
    assert X.shape[0] == lat.shape[0]
    rec = ae.predict(X)
    assert rec.shape == X.shape


def test_conditional_variational_autoencoder():
    ae = VariationalAutoencoder(
        x_dim = X.shape[1],
        conditional = 'all'
    )
    ae.compile(optimizer='adam', run_eagerly=True)
    check_attributes(ae, ce=True, cd=True, c=True)
    ae.fit(
        [X, cond],
        batch_size = 50,
        epochs = 2
    )
    lat = ae.transform([X, cond])
    assert X.shape[0] == lat.shape[0]
    rec = ae.predict([X, cond])
    assert rec.shape == X.shape


def test_negative_binomial_variational_autoencoder():
    ae = NegativeBinomialVAE(
        x_dim = X.shape[1],
        dispersion = 'cell-gene'
    )
    ae.compile(optimizer='adam', run_eagerly=False)
    check_attributes(ae, sf=True)
    ae.fit(
        [X, sf],
        batch_size = 50,
        epochs = 2
    )
    lat = ae.transform(X)
    assert X.shape[0] == lat.shape[0]
    rec = ae.predict([X, sf])
    assert rec.shape == X.shape
