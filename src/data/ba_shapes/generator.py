from src.data.ba_shapes.generator_motif import HouseMotif
import random
import networkx as nx
from typing import Tuple, Dict

class BAShapesGenerator:
    """Handles the creation of individual small BA graphs with or without motifs."""
    def __init__(self, num_base_nodes: int = 20, m_edges: int = 1):
        self.num_base_nodes = num_base_nodes
        self.m_edges = m_edges
        self.motif_generator = HouseMotif()

    def generate_sample(self, has_motif: bool) -> Tuple[nx.Graph, int]:
        if has_motif:
            n_base = self.num_base_nodes
            graph = nx.barabasi_albert_graph(n_base, self.m_edges)
            
            motif_graph, _ = self.motif_generator.generate_motif() 
            
            mapping = {n: n + n_base for n in motif_graph.nodes()}
            motif_graph = nx.relabel_nodes(motif_graph, mapping)
            
            graph.add_nodes_from(motif_graph.nodes())
            graph.add_edges_from(motif_graph.edges())
            
            base_connection_point = random.randint(0, n_base - 1)
            motif_connection_point = mapping[4] 
            graph.add_edge(base_connection_point, motif_connection_point) 
            
            label = 1
            
        else:
            n_base = self.num_base_nodes + 5
            graph = nx.barabasi_albert_graph(n_base, self.m_edges) 
            
            edges_added = 0
            while edges_added < 2:
                u = random.randint(0, n_base - 1)
                v = random.randint(0, n_base - 1)
                if u != v and not graph.has_edge(u, v):
                    graph.add_edge(u, v)
                    edges_added += 1
                    
            label = 0

        return graph, label