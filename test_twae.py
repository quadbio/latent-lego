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
from tensorflow.data import Dataset
from tensorflow.keras.losses import MeanSquaredError, Poisson
from keras.utils import plot_model

from latent.flow.twae import MMDTwinAutoencoder
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

    # n_umis = X_use.sum(1)
    # size_factors = n_umis / np.median(n_umis)

    rna_ae = Autoencoder(
        x_dim = rna_use.shape[1],
        activation = 'leaky_relu',
        latent_dim = 10
    )
    atac_ae = Autoencoder(
        x_dim = atac_use.shape[1],
        activation = 'leaky_relu',
        latent_dim = 10
    )
    twae = MMDTwinAutoencoder(
        models = [rna_ae, atac_ae],
        losses = [MeanSquaredError(), MeanSquaredError()]
    )
    twae.compile()
    history = twae.fit(
        [rna_use, atac_use],
        batch_size = args.batch_size,
        epochs = args.epochs,
        use_multiprocessing = True,
        workers = args.cpus
    )
    history.history

    rna_lat, atac_lat = twae.transform([rna_use, atac_use])

    adata.obsm['X_ae'] = latent
    sc.pp.neighbors(adata, use_rep='X_ae', n_neighbors=30)
    sc.tl.umap(adata, min_dist=0.1, spread=0.5)

    p = sc.pl.scatter(adata, show=False, basis='ae', color='celltype')
    p.figure.savefig('latent_final.png')

    p = sc.pl.scatter(adata, show=False, basis='umap', color='celltype')
    p.figure.savefig('umap_final.png')


    #### Custom model
    # rna_enc = Encoder(
    #     latent_dim = 10,
    #     name = 'rna_encoder'
    # )
    # rna_dec = Decoder(
    #     x_dim = rna_use.shape[1],
    #     name = 'rna_decoder'
    # )
    # atac_enc = Encoder(
    #     latent_dim = 10,
    #     name = 'atac_encoder'
    # )
    # atac_dec = Decoder(
    #     x_dim = atac_use.shape[1],
    #     name = 'atac_decoder'
    # )
    #
    # rna_input = layers.Input((rna_use.shape[1],))
    # atac_input = layers.Input((atac_use.shape[1],))
    #
    # latent1 = rna_enc(rna_input)
    # latent2 = atac_enc(atac_input)
    #
    # out1 = rna_dec(latent1)
    # out2 = atac_dec(latent2)
    #
    # model = Model(inputs=[rna_input, atac_input], outputs=[out1, out2])
    # model.compile(optimizer='adam', loss=None)
    # model.add_loss(losses.mean_squared_error(rna_input, out1))
    # model.add_loss(losses.mean_squared_error(atac_input, out2))
