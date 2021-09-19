import latent
import numpy as np
import scipy as sp
import tensorflow.keras as keras
from sklearn import preprocessing as pp
from scipy.spatial.distance import pdist

from typing import Iterable, Union
from latent.utils import aggregate, to_dense

# Perturbation prediction utils
def test_train_split(
    adata,
    celltype_key,
    predict_key,
    celltype_predict,
    control_name
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
    ctrl_cond = np.array(control_name)
    stim_idx = ~adata.obs.loc[:, predict_key].isin(ctrl_cond)
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
        model: latent.models.Autoencoder
    ):
        """
        Arguments:
            model: An autoencoder model.
        """
        self.model = model
        self.use_conditions = self.model._conditional_decoder()

    def predict(
        self, 
        adata,
        celltype_key: str,
        celltype_predict: str,
        predict_key: str,
        control_name: Union[str, Iterable[str]],
        condition_key: str = None,
        weighted: bool = False
    ):
        """
        Arguments:
            adata: An anndata object.
            celltype_key: String indicating the metadata column containing celltype info.
            celltype_predict: String indicating the cell type to predict.
            predict_key: String indicating the metadata column containing 
                the condition to predict info.
            control_name: String indicating the control condition.
            condition_key: String indicating the metadata column containing the 
                condition info (used for conditional autoencoders)
            weighted: Whether to weight the latent vectors.
        """
        ct_pred = np.array(celltype_predict)
        ctrl_cond = np.array(control_name)
        stim_idx = ~adata.obs.loc[:, predict_key].isin(ctrl_cond)
        celltypes = adata.obs[celltype_key]
        ct_idx = celltypes.isin(ct_pred)

        stim_pred_from = adata[(~ct_idx & stim_idx), :]
        ctrl_pred_from = adata[(~ct_idx & ~stim_idx), :]
        pred_to = adata[(ct_idx & ~stim_idx), :]
        
        latent_stim = to_dense(self.model.transform(stim_pred_from.X))
        latent_ctrl = to_dense(self.model.transform(ctrl_pred_from.X))
        latent_pred_to = to_dense(self.model.transform(pred_to.X))

        stim_mean = aggregate(
            latent_stim, groups=celltypes[(~ct_idx & stim_idx)], axis=0)
        ctrl_mean = aggregate(
            latent_ctrl, groups=celltypes[(~ct_idx & ~stim_idx)], axis=0)

        delta = stim_mean - ctrl_mean

        if weighted:
            weights = self._get_weights(
                latent_ctrl, celltypes[(~ct_idx & ~stim_idx)])
            mean_delta = np.mean(delta * weights)
        else:
            mean_delta = np.mean(delta)

        latent_pred = latent_pred_to + mean_delta

        if self.use_conditions:
            conditions = self._get_conditions(pred_to, condition_key)
            return self.model.reconstruct([latent_pred, conditions])
        else:
            return self.model.reconstruct(latent_pred)

    def _get_weights(self, latent, celltypes, metric='euclidean'):
        latent_mean = aggregate(latent, celltypes, axis=0)
        dists = sp.distance.pdist(latent_mean, metric=metric)
        rev_dists = 1 / (dists + 1)
        return rev_dists / np.sum(rev_dists)

    def _get_conditions(self, adata, condition_key):
        le = pp.LabelEncoder()
        cond = adata.obs[condition_key].values
        cond = le.fit_transform(cond)
        cond = keras.utils.to_categorical(cond)
