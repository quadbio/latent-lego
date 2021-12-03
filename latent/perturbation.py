import numpy as np
import scipy as sp
import tensorflow.keras as keras
from sklearn import preprocessing as pp
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
    group_pred = np.array(celltype_predict)
    pred_cond = np.array(predict_name)
    stim_idx = adata.obs.loc[:, predict_key].isin(pred_cond)
    ct_idx = adata.obs[celltype_key].isin(group_pred)
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

    def transform(self, adata, rep_name='latent', condition_key = None):
        """Get latent representation and store in AnnData object"""
        if self.use_conditions:
            conditions = np.array(adata.obs[condition_key])
            latent_space = self.model.transform(adata.X, conditions=conditions)
        else:
            latent_space = self.model.transform(adata.X)
        rep_name = 'X_' + rep_name
        adata.obsm[rep_name] = latent_space

    def get_latent_vectors(
        self,
        adata,
        group_key,
        predict_key: str,
        predict_label: Union[str, Iterable[str]],
        groups_use: Union[str, Iterable[str]] = None,
        use_rep: str = 'X_latent',
        return_vectors: bool = False
    ):
        """
        Calculate vectors in latent space.
        Arguments:
            adata: An AnnData object.
            group_key: String indicating the metadata column containing group info.
            predict_key: String indicating the metadata column containing 
                the condition to predict info.
            predict_label: String indicating the condition to predict.
            group_use: String indicating the cell type to use for calculating 
                latent vectors.
            use_rep: String indicating the metadata column containing the
                representation to use for calculating latent vectors.
            return_vectors: Boolean indicating whether to return the vectors.
        """
        pred_label = np.array(predict_label).tolist()
        stim_idx = adata.obs.loc[:, predict_key].isin([pred_label])
        groups = adata.obs[group_key]

        latent_rep = adata.obsm[use_rep]
        if groups_use is not None:
            groups_use = np.array(groups_use).tolist()
            groups_use_idx = groups.isin([groups_use])
            latent_rep = latent_rep[groups_use_idx, :]
            groups = groups[groups_use_idx]
            stim_idx = stim_idx[groups_use_idx]

        latent_stim = latent_rep[stim_idx, :]
        latent_ctrl = latent_rep[~stim_idx, :]

        stim_groups = groups[stim_idx].astype(str).values
        stim_mean = aggregate(latent_stim, groups=stim_groups, axis=0)
        ctrl_groups = groups[~stim_idx].astype(str).values
        ctrl_mean = aggregate(latent_ctrl, groups=ctrl_groups, axis=0)

        delta = stim_mean - ctrl_mean
        self.delta = delta
        if return_vectors:
            return delta


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
        group_pred = np.array(celltype_predict).tolist()
        pred_cond = np.array(predict_name).tolist()
        stim_idx = adata.obs.loc[:, predict_key].isin([pred_cond])
        celltypes = adata.obs[celltype_key]
        ct_idx = celltypes.isin([group_pred])

        stim_pred_from = adata[(~ct_idx & stim_idx), :]
        ctrl_pred_from = adata[(~ct_idx & ~stim_idx), :]
        ctrl_all = adata[(~stim_idx), :]
        pred_to = adata[(ct_idx & ~stim_idx), :]
        
        latent_stim = to_dense(self.model.transform(stim_pred_from.X))
        latent_ctrl = to_dense(self.model.transform(ctrl_pred_from.X))
        latent_all = to_dense(self.model.transform(ctrl_all.X))
        latent_pred_to = to_dense(self.model.transform(pred_to.X))

        stim_groups = celltypes[(~ct_idx & stim_idx)].astype(str).values
        stim_mean = aggregate(latent_stim, groups=stim_groups, axis=0)
        ctrl_groups = celltypes[(~ct_idx & ~stim_idx)].astype(str).values
        ctrl_mean = aggregate(latent_ctrl, groups=ctrl_groups, axis=0)

        delta = stim_mean - ctrl_mean

        if weighted:
            ctrl_groups_all = celltypes[~stim_idx].astype(str).values
            predict_idx = celltypes[~stim_idx].isin([group_pred]).values
            weights = self._get_weights(
                latent=latent_all,
                groups=ctrl_groups_all,
                predict_idx=predict_idx,
                metric=metric
            )
            mean_delta = np.sum(delta * weights, axis=0)
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

    def _get_weights(self, latent, groups, predict_idx, metric='euclidean'):
        from_mean = aggregate(latent[~predict_idx], groups[~predict_idx], axis=0)
        to_mean = aggregate(latent[predict_idx], groups[predict_idx], axis=0)
        dists = sp.spatial.distance.cdist(from_mean, to_mean, metric=metric)
        rev_dists = 1 / (dists + 1)
        return rev_dists / np.sum(rev_dists)

    def _get_conditions(self, adata, condition_key):
        le = pp.LabelEncoder()
        cond = adata.obs[condition_key].values
        cond = le.fit_transform(cond)
        cond = keras.utils.to_categorical(cond)
        return cond
