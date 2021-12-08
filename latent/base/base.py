"""Base class for autoencoder models"""

import pickle
import inspect
import numpy as np
from abc import ABC

class BaseModel(ABC):
    """Abstract base class for autoencoder models"""

    def __init__(self):
        """Initialize base model"""
        pass

    def transform(self):
        """Transform input to latent representation"""
        raise NotImplementedError

    def reconstruct(self):
        """Reconstruct input from latent representation"""
        raise NotImplementedError

    def encode(self):
        """Encode input"""
        raise NotImplementedError

    def decode(self):
        """Decode latent"""
        raise NotImplementedError

    def save(self, filename):
        """Save model"""
        init_params = self._get_init_params()
        weights = self.get_weights() 
        model_dict = dict(
            init_params=init_params,
            weights=weights
        )
        with open(filename, 'wb') as f:
            pickle.dump(model_dict, f)

    def build_model(self, input):
        """Build model"""
        self.__call__(input)
        
    @classmethod
    def load(self, filename, input):
        """Load model"""
        with open(filename, 'rb') as f:
            model_dict = pickle.load(f)
        init_params = model_dict['init_params']
        weights = model_dict['weights']
        self.__init__(**init_params['args'], **init_params['kwargs'])
        self.build_model(input)
        self.set_weights(weights)

    def _get_init_params(self):
        """Retruns the parameters needed for reinitialization of the same model"""
        init = self.__init__
        sig = inspect.signature(init)
        exclude_params = ['encoder', 'decoder', 'kwargs']
        init_params = [p for p in sig.parameters.keys()
                       if p not in exclude_params]
        args_dict = {}
        for p in init_params:
            args_dict[p] = getattr(self, p)
        init_dict = {}
        init_dict['args'] = args_dict
        init_dict['kwargs'] = self.net_kwargs
        return init_dict
