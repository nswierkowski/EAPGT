import torch
import copy
from torch_geometric.data import Data
from .base import CounterfactualEngine

class BAShapesCounterfactual(CounterfactualEngine):
    def generate(self, data: Data) -> Data:
        corrupted = copy.deepcopy(data)
        
        if data.y.item() == 1:
            num_nodes = data.num_nodes
            keep_nodes = num_nodes - 5
            
            corrupted.x = data.x[:keep_nodes]
            
            edge_index = data.edge_index
            mask = (edge_index[0] < keep_nodes) & (edge_index[1] < keep_nodes)
            corrupted.edge_index = edge_index[:, mask]
            
            corrupted.y = torch.tensor([0], dtype=torch.long)
            
        else:
            num_nodes = data.num_nodes
            house_edges = torch.tensor([
                [0, 1], [1, 2], [2, 0], 
                [1, 3], [2, 4], [3, 4]  
            ], dtype=torch.long).t()
            
            house_edges += num_nodes
            
            connection = torch.tensor([[0, num_nodes]], dtype=torch.long).t()
            
            corrupted.edge_index = torch.cat([data.edge_index, house_edges, connection], dim=1)
            
            new_features = torch.zeros((5, data.x.size(1))) 
            corrupted.x = torch.cat([data.x, new_features], dim=0)
            
            corrupted.y = torch.tensor([1], dtype=torch.long)

        return corrupted