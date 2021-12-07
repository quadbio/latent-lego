import inspect
import numpy as np
from typing import Literal
from latent.models import Autoencoder

class ArchitectureSurgery:
    """Architecture surgery for transfer learning using the 
    scArches algorithm [Lotfollahi21]
    """

    # This is a keras implementation of the scArches approach here: 
    # https://github.com/YosefLab/scvi-tools/blob/master/scvi/model/base/_archesmixin.py
    def update_architecture(
        self,
        model: Autoencoder,
        add_condition: Literal['first', 'all'] = 'first',
        unfrozen: bool = False,
        freeze_dropout: bool = False,
        freeze_batchnorm_encoder: bool = False,
        freeze_batchnorm_decoder: bool = False,
        freeze_weights: dict = None 
    ):
        """Update the architecture of the model. Adds condidtion weights for the query 
        data and freezes other layers for training.

        Arguments: 
            model: The model to update.
            add_condition: Whether to add the condition weights to first 
                or all layers of the model.
            unfrozen: Whether to not freeze the model. Overwrites all other 
                freezing arguments. 
            freeze_dropout: Whether to freeze dropout layers.
            freeze_batchnorm_encoder: Whether to freeze batchnorm layers in the 
                encoder.
            freeze_batchnorm_decoder: Whether to freeze batchnorm layers in the
                decoder.
            freeze_weights: A dictionary of layer names to freeze weights for.
        """

        # Clone model
        if freeze_dropout:
            new_model = self.clone_model(model, conditional=add_condition, dropout_rate=0)
        else:
            new_model = self.clone_model(model, conditional=add_condition)

        # Get old input shape
        old_x_shape = model.encoder.layers[0]._saved_model_inputs_spec[0].shape[-1]

        # Get old condition shape
        old_conditional = model._conditional_encoder() or model._conditional_encoder()
        if old_conditional:
            if model._conditional_encoder():
                old_cond_shape = (model.encoder.layers[0]
                    ._saved_model_inputs_spec[1].shape[-1])
            else:
                old_cond_shape = (model.decoder.layers[0]
                    ._saved_model_inputs_spec[1].shape[-1])
            # Initiate new condtion shape
            new_cond_shape = old_cond_shape + 1

        else:
            new_cond_shape = 1

        # Add size factor inputs
        if model._use_sf():
            new_input_shape = [(1,old_x_shape), (1,new_cond_shape), (1,1)]
        else:
            new_input_shape = [(1,old_x_shape), (1,new_cond_shape)]
        # Make dummy input with new input shapes
        dummy_input = [np.ones(s) for s in new_input_shape]
        # Initialize model weights with new shapes
        _ = new_model(dummy_input)

        # Iterate through weights and adapt shape
        new_weights = []
        for new_w, old_w in zip(new_model.weights, model.weights):
            if new_w.shape == old_w.shape:
                new_weights.append(old_w.numpy())
                continue
            else:
                dim_diff = new_w.shape[0] - old_w.shape[0]
                new_w, old_w = (new_w.numpy(), old_w.numpy())
                new_w_updated = np.concatenate([old_w, new_w[-dim_diff:, :]], axis=0)
                new_weights.append(new_w_updated)
        
        # Update model weights
        new_model.set_weights(new_weights)

        # Freeze layers if requested
        if unfrozen:
            return new_model
        else:
            new_model = self.freeze_layers(
                new_model, 
                freeze_dropout=freeze_dropout, 
                freeze_batchnorm_encoder=freeze_batchnorm_encoder, 
                freeze_batchnorm_decoder=freeze_batchnorm_decoder, 
                freeze_weights=freeze_weights
            )
            return new_model







    def clone_model(self, model, **kwargs):
        """Reinstantiate an autoencoder model with empty weights."""
        model_params = self._get_init_params(model)
        encoder_params = self._get_init_params(model.encoder)
        decoder_params = self._get_init_params(model.decoder)

        # Update kwargs with the new model parameters
        model_params['kwargs'] = self._set_params(model_params['kwargs'], kwargs)
        encoder_params['kwargs'] = self._set_params(encoder_params['kwargs'], kwargs)
        decoder_params['kwargs'] = self._set_params(decoder_params['kwargs'], kwargs)

        # Create new model
        encoder = model.encoder.__class__(
            **encoder_params['args'], 
            **encoder_params['kwargs']
        )
        decoder = model.decoder.__class__(
            **decoder_params['args'], 
            **decoder_params['kwargs']
        )
        new_model = model.__class__(
            encoder=encoder, 
            decoder=decoder, 
            **model_params['args'],
            **model_params['kwargs']
        )

        return new_model

    def _get_init_params(self, model):
        """Retruns the parameters needed for reinitialization of the same model"""
        init = model.__init__
        sig = inspect.signature(init)
        exclude_params = ['encoder', 'decoder', 'kwargs']
        init_params = [p for p in sig.parameters.keys()
                       if p not in exclude_params]
        args_dict = {}
        for p in init_params:
            args_dict[p] = getattr(model, p)
        init_dict = {}
        init_dict['args'] = args_dict
        init_dict['kwargs'] = model.net_kwargs
        return init_dict

    def _set_params(self, old_kwargs, new_kwargs):
        """Updates kwarg dict"""
        for k, v in new_kwargs.items():
            old_kwargs[k] = v
        return old_kwargs
