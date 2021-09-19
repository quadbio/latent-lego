import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


# Perturbation prediction utils
def test_train_split(
    adata,
    celltype_key,
    condition_key,
    celltype_predict,
    control_condition
):
    """Splits andata object into training and test set for pertrubation prediction. 
    The training set consists of controls of all cell types and stimulated state for all 
    but one cell type. The exluded test data will further be used as a ground truth for 
    evaluation of the perturbation prediction.

    Arguments:
        adata: An anndata object.
        celltype_predict: String indicating the metadata column containing celltype info.
        condition_key: String indicating the metadata column containing condition info.
        celltype_predict: A string or list of strings containing the celltype(s) 
            to be held out from training.
        control_condition: A string or list of strings containing the control conditions.
    """
    ct_pred = np.array(celltype_predict)
    ctrl_cond = np.array(control_condition)
    stim_idx = ~adata.obs.loc[:, condition_key].isin(ctrl_cond)
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
        model,
        adata,
        celltype_key,
        celltype_predict,
        condition_key,
        control_condition,
        weighted = False
    ):
        """
        Arguments:
            model: A trained autoencoder model.
            adata: An anndata object.
            celltype_key: String indicating the metadata column containing celltype info.
            condition_key: String indicating the metadata column containing 
                condition info.
            celltye_weights: Whether to weight the latent vectors.
        """
