import torch
import networkx as nx
from torch_geometric.utils import to_networkx, degree

class GraphormerTransform:
    """
    Calculates shortest paths and degrees for Graphormer.
    Apply this to your datasets before training!
    """
    def __init__(self, config=20):
        if isinstance(config, dict):
            self.max_spatial_pos = config.get('spatial_pos_max', 20)
        else:
            self.max_spatial_pos = config

    def __call__(self, data):
        row, col = data.edge_index
        data.in_degree = degree(col, data.num_nodes, dtype=torch.long)
        data.out_degree = degree(row, data.num_nodes, dtype=torch.long)

        nx_graph = to_networkx(data, to_undirected=True)
        shortest_paths = dict(nx.all_pairs_shortest_path_length(nx_graph))
        
        spatial_pos = torch.full((data.num_nodes, data.num_nodes), 
                                 fill_value=self.max_spatial_pos + 1, 
                                 dtype=torch.long)
        
        for i in range(data.num_nodes):
            for j in range(data.num_nodes):
                if i in shortest_paths and j in shortest_paths[i]:
                    dist = shortest_paths[i][j]
                    spatial_pos[i, j] = min(dist, self.max_spatial_pos)
                    
        data.spatial_pos = spatial_pos
        return data