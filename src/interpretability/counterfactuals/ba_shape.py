import torch
import copy
import random
from torch_geometric.data import Data
from .base import CounterfactualEngine

class BAShapesCounterfactual(CounterfactualEngine):
    def generate(self, data: Data) -> Data:
        corrupted = copy.deepcopy(data)
        
        if data.y.item() == 1:
            num_nodes = data.num_nodes
            base_nodes = num_nodes - 5
            
            m_nodes = list(range(base_nodes, num_nodes))
            
            mask = (data.edge_index[0] >= base_nodes) & (data.edge_index[1] >= base_nodes)
            
            new_edges = []
            
            center_node = m_nodes[0]
            for leaf_node in m_nodes[1:]:
                new_edges.append([center_node, leaf_node])
                new_edges.append([leaf_node, center_node]) 
            
            existing_edges = set(map(tuple, data.edge_index.t().tolist()))
            added_anchors = 0
            
            while added_anchors < 2:
                u = random.choice(m_nodes)
                v = random.randint(0, base_nodes - 1)
                
                if (u, v) not in existing_edges and (v, u) not in existing_edges:
                    new_edges.append([u, v])
                    new_edges.append([v, u]) 
                    existing_edges.add((u, v))
                    existing_edges.add((v, u))
                    added_anchors += 1
                    
            new_edge_tensor = torch.tensor(new_edges, dtype=torch.long).t().to(data.edge_index.device)
            corrupted.edge_index[:, mask] = new_edge_tensor
            
            corrupted.y = torch.tensor([0], dtype=torch.long)
            
        return corrupted