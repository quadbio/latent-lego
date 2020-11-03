import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import scanpy as sc
import anndata as ad

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from tensorflow.keras import backend as K
from tensorflow.data import Dataset
from tensorflow.keras.losses import MeanSquaredError, Poisson
from keras.utils import plot_model

from latent.flow.twae import TwinAutoencoder
from latent.flow.ae import Autoencoder, PoissonAutoencoder
from latent.flow.vae import NegativeBinomialVAE as NBVAE
from latent.flow.ae import NegativeBinomialAutoencoder as NBAE
from latent.flow.encoder import Encoder
from latent.flow.decoder import Decoder


# FUNC
def interface():
    parser = argparse.ArgumentParser(description='Fits autoencoder')

    parser.add_argument('H5AD',
                        type=str,
                        nargs=2,
                        metavar='<file>',
                        help='Input H5AD or LOOM file.')

    parser.add_argument('-e', '--epochs',
                        type=int,
                        dest='epochs',
                        default='20',
                        metavar='<int>',
                        help='Number of epochs to train.')

    parser.add_argument('-b', '--batch-size',
                        type=int,
                        dest='batch_size',
                        default='100',
                        metavar='<int>',
                        help='Number of epochs to train.')

    parser.add_argument('-c', '--cpus',
                        type=int,
                        dest='cpus',
                        default='5',
                        metavar='<int>',
                        help='Number of cores to use.')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = interface()

    rna = sc.read(args.H5AD[0])
    atac = sc.read(args.H5AD[1])
    rna_use = np.array(rna.X.todense())[:5000, :]
    atac_use = np.array(atac.X.todense())[:5000, :]

    n_umis = rna_use.sum(1)
    rna_sf = n_umis / np.median(n_umis)

    n_counts = atac_use.sum(1)
    atac_sf = n_counts / np.median(n_counts)

    rna_ae = NBAE(
        x_dim = rna_use.shape[1],
        activation = 'leaky_relu',
        latent_dim = 20
    )
    atac_ae = NBAE(
        x_dim = atac_use.shape[1],
        activation = 'leaky_relu',
        latent_dim = 20
    )
    twae = TwinAutoencoder([rna_ae, atac_ae])
    twae.compile()
    history = twae.fit(
        [[rna_use, rna_sf], [atac_use, atac_sf]],
        batch_size = args.batch_size,
        epochs = args.epochs,
        use_multiprocessing = True,
        workers = args.cpus
    )

    latent, labels = twae.transform([rna_use, atac_use])

    latad = ad.AnnData(X=latent)
    latad.tech = labels
    adata.obsm['X_X'] = latent

    sc.pp.neighbors(latad, use_rep='X', n_neighbors=30)
    sc.tl.umap(latad, min_dist=0.1, spread=0.5)

    p = sc.pl.scatter(latad, show=False, basis='X', color='tech')
    p.figure.savefig('latent_twae.png')

    p = sc.pl.scatter(latad, show=False, basis='umap', color='tech')
    p.figure.savefig('umap_twae.png')
