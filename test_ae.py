import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import scanpy as sc

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from tensorflow.keras import backend as K
from keras.losses import mean_squared_error
from keras.utils import plot_model

from latent.flow.ae import Autoencoder, PoissonAutoencoder
from latent.flow.ae import NegativeBinomialAutoencoder as NBAE
from latent.flow.ae import ZINBAutoencoder as ZINBAE

from latent.flow.vae import VariationalAutoencoder, ZINBVAE
from latent.flow.vae import NegativeBinomialVAE as NBVAE

# FUNC
def interface():
    parser = argparse.ArgumentParser(description='Fits autoencoder')

    parser.add_argument('H5AD',
                        type=str,
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

    # adata = sc.datasets.paul15()
    adata = sc.read(args.H5AD)
    # adata.X = adata.raw.X
    # sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3')
    # adata = adata[:,adata.var.highly_variable]
    X_use = np.array(adata.X.todense())

    n_umis = X_use.sum(1)
    size_factors = n_umis / np.median(n_umis)

    autoencoder = NBAE(
        x_dim = X_use.shape[1],
        # beta = 1e-3,
        activation = 'prelu',
        latent_dim = 10,
        hidden_units = [256, 128]
    )
    autoencoder.compile()
    autoencoder.fit(
        [X_use, size_factors],
        batch_size = args.batch_size,
        epochs = args.epochs,
        use_multiprocessing = True,
        workers = args.cpus
    )

    # autoencoder = Autoencoder(
    #     x_dim = X_use.shape[1],
    #     latent_dim = 20
    # )
    # autoencoder.compile()
    # autoencoder.fit(
    #     X_use,
    #     batch_size = args.batch_size,
    #     epochs = args.epochs,
    #     use_multiprocessing = True,
    #     workers = args.cpus
    # )

    latent = autoencoder.transform(X_use)
    adata.obsm['X_ae'] = latent
    sc.pp.neighbors(adata, use_rep='X_ae', n_neighbors=30)
    sc.tl.umap(adata, min_dist=0.1, spread=0.5)

    p = sc.pl.scatter(adata, show=False, basis='ae', color='celltype')
    p.figure.savefig('latent_final.png')

    p = sc.pl.scatter(adata, show=False, basis='umap', color='celltype')
    p.figure.savefig('umap_final.png')
