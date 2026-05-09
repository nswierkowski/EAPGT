import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseMessageModule(nn.Module, ABC):
    """Abstract base class for the Message step of an MPNN."""
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

class BaseAggregateModule(nn.Module, ABC):
    """Abstract base class for the Aggregate step of an MPNN."""
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

class BaseUpdateModule(nn.Module, ABC):
    """Abstract base class for the Update step of an MPNN."""
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

class BaseMPNNWrapper(nn.Module, ABC):
    """
    Abstract base class for all MPNN wrappers.
    Enforces that every wrapped MPNN exposes a Message, Aggregate, and Update module
    so that NNsight and EAP strategies can interface with them uniformly.
    """
    def __init__(
        self, 
        message_module: BaseMessageModule, 
        aggregate_module: BaseAggregateModule, 
        update_module: BaseUpdateModule
    ):
        super().__init__()
        self.message_module = message_module
        self.aggregate_module = aggregate_module
        self.update_module = update_module

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        The forward pass must define how the data flows through the 
        message, aggregate, and update modules.
        """
        pass