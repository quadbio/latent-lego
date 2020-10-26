import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import scanpy as sc

from keras.losses import mean_squared_error
from keras.utils import plot_model

from latent.flow.ae import Autoencoder, CountAutoencoder, PoissonAutoencoder

# FUNC
def interface():
    parser = argparse.ArgumentParser(description='Fits autoencoder')

    # parser.add_argument('H5AD',
    #                     type=str,
    #                     metavar='<file>',
    #                     help='Input H5AD or LOOM file.')

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

    adata = sc.datasets.paul15()
    X_use = adata.X
    n_umis = X_use.sum(1)
    size_factors = n_umis / np.median(n_umis)

    n = X_use.shape[0]
    batch_num = int(n // args.batch_size)
    m = batch_num * args.batch_size

    autoencoder = PoissonAutoencoder(
        x_dim = X_use.shape[1],
        latent_dim = 20
    )
    autoencoder.fit(
        [X_use, size_factors],
        batch_size = args.batch_size,
        epochs = args.epochs,
        use_multiprocessing = True,
        workers = args.cpus
    )

    latent = autoencoder.transform(X_use)
    adata.obsm['X_ae'] = latent
    sc.pp.neighbors(adata, use_rep='X_ae')
    sc.tl.umap(adata)

    p = sc.pl.scatter(adata, show=False, basis='ae', color='paul15_clusters')
    p.figure.savefig('latent_final.png')

    p = sc.pl.scatter(adata, show=False, basis='umap', color='paul15_clusters')
    p.figure.savefig('umap_final.png')
