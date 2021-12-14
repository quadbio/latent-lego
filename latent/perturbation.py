import numpy as np
import scipy as sp
import tensorflow.keras as keras
from sklearn import preprocessing as pp
import anndata as ad

from typing import Iterable, Union
from latent.models import Autoencoder
from latent.utils import aggregate, to_dense, get_conditions

# Perturbation prediction utils
def test_train_split(adata, celltype_key, predict_key, celltype_predict, predict_name):
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

    def __init__(self, model: Autoencoder):
        """
        Arguments:
            model: An autoencoder model.
        """
        self.model = model
        self.use_conditions = self.model._conditional_decoder()
        self.use_sf = self.model._use_sf()

    def transform(self, adata, rep_name="latent", condition_key=None):
        """Get latent representation and store in AnnData object"""
        if self.use_conditions:
            conditions = self._get_conditions(adata, condition_key)
            latent_space = self.model.transform(adata.X, conditions=conditions)
        else:
            latent_space = self.model.transform(adata.X)
        rep_name = "X_" + rep_name
        adata.obsm[rep_name] = latent_space

    def predict(
        self,
        adata,
        group_key: str,
        groups_predict: Union[str, Iterable[str]],
        predict_key: str = None,
        predict_label: Union[str, Iterable[str]] = None,
        condition_key: str = None,
        use_rep: str = "X_latent",
        weighted: bool = False,
        weight_rep: str = "X_pca",
        metric: str = "euclidean",
        return_adata: bool = False,
    ):
        """
        Calculate latent vectors given an latent representstion and predict perturbation
        by adding them in latent space.
        Arguments:
            adata: An anndata object.
            group_key: String indicating the metadata column containing group info.
            groups_predict: String indicating the group to predict f.
            predict_key: String indicating the metadata column containing
                the condition to predict info.
            predict_name: String indicating the condition to predict.
            condition_key: String indicating the metadata column containing the
                condition info (used for conditional autoencoders)
            size_factor_key: String indicating the metadata column containing the
                size factors.
            weighted: Whether to weight the latent vectors.
            metric: String indicating the metric to use for distance calculation.
            use_rep: String indicating the metadata column containing the
                representation to use for calculating latent vectors.
            return_adata: Whether to return the perturbed adata object.
        """
        group_pred = np.array(groups_predict).tolist()
        pred_label = np.array(predict_label).tolist()
        stim_idx = adata.obs.loc[:, predict_key].isin([pred_label])
        groups = adata.obs[group_key]
        group_idx = groups.isin([group_pred])

        # Get intersection groups between stim and non-stim
        intersect_groups = np.intersect1d(groups[stim_idx], groups[~stim_idx])
        intersect_idx = groups.isin(intersect_groups)
        group_train_idx = intersect_idx & ~group_idx

        latent_rep = adata.obsm[use_rep]

        latent_stim = latent_rep[(group_train_idx & stim_idx), :]
        latent_ctrl = latent_rep[(group_train_idx & ~stim_idx), :]
        latent_pred_to = latent_rep[(group_idx & ~stim_idx), :]
        adata_pred_to = adata[(group_idx & ~stim_idx), :]

        stim_groups = groups[(group_train_idx & stim_idx)].astype(str).values
        stim_mean = aggregate(latent_stim, groups=stim_groups, axis=0)
        ctrl_groups = groups[(group_train_idx & ~stim_idx)].astype(str).values
        ctrl_mean = aggregate(latent_ctrl, groups=ctrl_groups, axis=0)

        delta = stim_mean - ctrl_mean

        if weighted:
            try:
                weight_rep = adata.X if weight_rep == "X" else adata.obsm[weight_rep]
            except KeyError:
                raise KeyError(f"The representation {weight_rep} does not exist.")
            weight_idx = (intersect_idx | group_idx) & ~stim_idx
            weight_rep = to_dense(weight_rep)[(weight_idx), :]
            ctrl_groups_all = groups[weight_idx].astype(str).values
            predict_idx = groups[weight_idx].isin([group_pred]).values
            weights = self._get_weights(
                rep=weight_rep,
                groups=ctrl_groups_all,
                predict_idx=predict_idx,
                metric=metric,
            )
            mean_delta = np.sum(delta * weights, axis=0)
            self.weights = np.squeeze(weights)
        else:
            mean_delta = np.mean(delta, axis=0)
            self.weights = np.array([1.0] * delta.shape[0])

        latent_pred = latent_pred_to + mean_delta

        if self.use_conditions:
            conditions = self._get_conditions(adata, condition_key)
            conditions = conditions[(group_idx & ~stim_idx), :]
            x_pred = self.model.reconstruct(latent_pred, conditions=conditions)
        else:
            x_pred = self.model.reconstruct(latent_pred)

        self.delta = delta
        self.mean_delta = mean_delta

        if return_adata:
            ad_pred = ad.AnnData(
                X=x_pred, obs=adata_pred_to.obs.copy(), var=adata_pred_to.var.copy()
            )
            ad_pred.obs[predict_key] = "PREDICTED"
            return ad_pred
        else:
            return x_pred

    def _get_weights(self, rep, groups, predict_idx, metric="euclidean"):
        from_mean = aggregate(rep[~predict_idx], groups[~predict_idx], axis=0)
        to_mean = aggregate(rep[predict_idx], groups[predict_idx], axis=0)
        dists = sp.spatial.distance.cdist(from_mean, to_mean, metric=metric)
        rev_dists = 1 / (dists + 1)
        return rev_dists / np.sum(rev_dists)

    def _get_conditions(self, adata, condition_key):
        le = pp.LabelEncoder()
        cond = adata.obs[condition_key].values
        cond = le.fit_transform(cond)
        cond = keras.utils.to_categorical(cond)
        return cond
