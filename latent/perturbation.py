import numpy as np
import scipy as sp
import tensorflow.keras as keras
from sklearn import preprocessing as pp
from scipy.spatial.distance import pdist
import anndata as ad

from typing import Iterable, Union
from latent.models import Autoencoder
from latent.utils import aggregate, to_dense

# Perturbation prediction utils
def test_train_split(
    adata,
    celltype_key,
    predict_key,
    celltype_predict,
    predict_name
):
    """Splits andata object into training and test set for pertrubation prediction. 
    The training set consists of controls of all cell types and stimulated state for all 
    but one cell type. The exluded test data will further be used as a ground truth for 
    evaluation of the perturbation prediction.

    Arguments:
        adata: An anndata object.
        celltype_predict: String indicating the metadata column containing celltype info.
        predict_key: String indicating the metadata column containing condition info.
        celltype_predict: A string or list of strings containing the celltype(s) 
            to be held out from training.
        control_name: A string or list of strings containing the control conditions.
    """
    ct_pred = np.array(celltype_predict)
    pred_cond = np.array(predict_name)
    stim_idx = adata.obs.loc[:, predict_key].isin(pred_cond)
    ct_idx = adata.obs[celltype_key].isin(ct_pred)
    train = adata[~(stim_idx & ct_idx), :]
    test = adata[~(stim_idx & ct_idx), :]
    return test, train


class LatentVectorArithmetics:
    """Pertrubation predictions through vector arithmetics in latent space 
    and reconstruction of the result. 
    """
    def __init__(
        self,
        model: Autoencoder
    ):
        """
        Arguments:
            model: An autoencoder model.
        """
        self.model = model
        self.use_conditions = self.model._conditional_decoder()
        self.use_sf = self.model._use_sf()

    def predict(
        self, 
        adata,
        celltype_key: str,
        celltype_predict: str,
        predict_key: str,
        predict_name: Union[str, Iterable[str]],
        condition_key: str = None,
        weighted: bool = False,
        metric: str = 'euclidean',
        return_adata: bool = False
    ):
        """
        Arguments:
            adata: An anndata object.
            celltype_key: String indicating the metadata column containing celltype info.
            celltype_predict: String indicating the cell type to predict.
            predict_key: String indicating the metadata column containing 
                the condition to predict info.
            predict_name: String indicating the condition to predict.
            condition_key: String indicating the metadata column containing the 
                condition info (used for conditional autoencoders)
            size_factor_key: String indicating the metadata column containing the 
                size factors.
            weighted: Whether to weight the latent vectors.
            metric: String indicating the metric to use for distance calculation.
            return_adata: Whether to return the perturbed adata object.
        """
        ct_pred = np.array(celltype_predict).tolist()
        pred_cond = np.array(predict_name).tolist()
        stim_idx = adata.obs.loc[:, predict_key].isin([pred_cond])
        celltypes = adata.obs[celltype_key]
        ct_idx = celltypes.isin([ct_pred])

        stim_pred_from = adata[(~ct_idx & stim_idx), :]
        ctrl_pred_from = adata[(~ct_idx & ~stim_idx), :]
        pred_to = adata[(ct_idx & ~stim_idx), :]
        
        latent_stim = to_dense(self.model.transform(stim_pred_from.X))
        latent_ctrl = to_dense(self.model.transform(ctrl_pred_from.X))
        latent_pred_to = to_dense(self.model.transform(pred_to.X))

        stim_groups = celltypes[(~ct_idx & stim_idx)].astype(str).values
        stim_mean = aggregate(latent_stim, groups=stim_groups, axis=0)
        ctrl_groups = celltypes[(~ct_idx & ~stim_idx)].astype(str).values
        ctrl_mean = aggregate(latent_ctrl, groups=ctrl_groups, axis=0)

        delta = stim_mean - ctrl_mean

        if weighted:
            weights = self._get_weights(
                latent_ctrl, celltypes[(~ct_idx & ~stim_idx)], 
                metric=metric
            )
            mean_delta = np.mean(delta * weights, axis=0)
        else:
            mean_delta = np.mean(delta, axis=0)

        latent_pred = latent_pred_to + mean_delta

        if self.use_conditions:
            conditions = self._get_conditions(pred_to, condition_key)
            x_pred = self.model.reconstruct([latent_pred, conditions])
        else:
            x_pred = self.model.reconstruct(latent_pred)

        if return_adata:
            ad_pred = ad.AnnData(
                X=x_pred, 
                obs=pred_to.obs.copy(),
                var=pred_to.var.copy()
            )
            ad_pred.obs[predict_key] = 'PREDICTED'
            return ad_pred
        else:
            return x_pred

    def _get_weights(self, latent, celltypes, metric='euclidean'):
        latent_mean = aggregate(latent, celltypes, axis=0)
        dists = sp.spatial.distance.pdist(latent_mean, metric=metric)
        rev_dists = 1 / (dists + 1)
        return rev_dists / np.sum(rev_dists)

    def _get_conditions(self, adata, condition_key):
        le = pp.LabelEncoder()
        cond = adata.obs[condition_key].values
        cond = le.fit_transform(cond)
        cond = keras.utils.to_categorical(cond)
