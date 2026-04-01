from abc import ABC, abstractmethod
import torch
from torch_geometric.data import Data

class CounterfactualEngine(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def generate(self, data: Data) -> Data:
        """Returns a corrupted version of the input graph."""
        pass