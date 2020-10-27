"""Base class for pytorch Autoencoder"""

import abc

import torch.nn as nn

class AE(nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for autoencoder model"""

    @abc.abstractmethod
    def forward(self, x)
        """Compute loss for model."""

    @abc.abstractmethod
    def encode(self, x):
        """Compute latent representation."""

    @abc.abstractmethod
    def decode(self, z):
        """Compute reconstruction."""
