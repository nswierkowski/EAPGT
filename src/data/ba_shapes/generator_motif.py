import networkx as nx
from typing import Tuple, Dict
from abc import ABC, abstractmethod


class MotifGenerator(ABC):
    """Abstract base class for generating graph motifs."""
    @abstractmethod
    def generate_motif(self) -> Tuple[nx.Graph, Dict[int, int]]:
        pass

class HouseMotif(MotifGenerator):
    """
    Generates a 5-node house motif (triangle on top of a square).
    Roles: 1 (top), 2 (middle), 3 (bottom).
    """
    def generate_motif(self) -> Tuple[nx.Graph, Dict[int, int]]:
        house = nx.Graph()
        
        # Add edges for the house (0 is top, 1 & 2 are middle, 3 & 4 are bottom)
        edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]
        house.add_edges_from(edges)
        
        roles = {0: 1, 1: 2, 2: 2, 3: 3, 4: 3}
        return house, roles
