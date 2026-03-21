import torch.nn as nn
from abc import ABC, abstractmethod

class BaseGraphTransformer(nn.Module, ABC):
    """
    Abstract base class for all Graph Transformers in the project.
    Enforces a strict API for training and Mechanistic Interpretability.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, batch):
        """
        Standard PyTorch forward pass.
        Expects a PyTorch Geometric Batch object.
        """
        pass

    @abstractmethod
    def get_patchable_components(self):
        """
        CRITICAL FOR EAP: Every model must explicitly return a dictionary 
        of its patchable layers (e.g., attention weights, message passing edges, 
        or spatial biases) so the EAP hooks know exactly where to attach.
        """
        pass