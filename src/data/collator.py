import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch

class GraphTransformerCollator:
    """
    Routes the batch to standard PyG batching (for GraphGPS) 
    or dense tensor padding (for Graphormer).
    """
    def __init__(self, config, pad_token_id=0):
        self.model_name = config.get('model', {}).get('name', 'graphormer').lower()
        self.pad_token_id = pad_token_id
        
    def __graphgps_collate(self, batch):
        for data in batch:
            keys_to_remove = []
            for k, v in data.items():
                if isinstance(v, torch.Tensor) and v.dim() == 2:

                    if v.size(0) == v.size(1) and v.size(0) == data.num_nodes:
                        keys_to_remove.append(k)

                elif isinstance(v, torch.Tensor) and v.dim() == 3 and k == 'edge_input':
                        keys_to_remove.append(k)
            
            for k in keys_to_remove:
                if hasattr(data, k):
                    delattr(data, k)
                    
        return Batch.from_data_list(batch)
    
    def __graphormer_collate(self, batch):
        node_features = [data.x for data in batch]
        input_nodes = pad_sequence(node_features, batch_first=True, padding_value=self.pad_token_id)
        
        in_degrees = [data.in_degree for data in batch]
        out_degrees = [data.out_degree for data in batch]
        in_degree = pad_sequence(in_degrees, batch_first=True, padding_value=0)
        out_degree = pad_sequence(out_degrees, batch_first=True, padding_value=0)

        batch_size = len(batch)
        max_nodes = input_nodes.size(1)
        
        spatial_pos = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.long)
        attn_bias = torch.zeros((batch_size, max_nodes + 1, max_nodes + 1), dtype=torch.float)
        
        for i, data in enumerate(batch):
            n = data.num_nodes
            spatial_pos[i, :n, :n] = data.spatial_pos
            
            bias_matrix = torch.zeros((n + 1, n + 1), dtype=torch.float)
            
            full_bias = torch.full((max_nodes + 1, max_nodes + 1), -1e9, dtype=torch.float)
            full_bias[:n+1, :n+1] = bias_matrix
            attn_bias[i] = full_bias

        return {
            "input_nodes": input_nodes,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "spatial_pos": spatial_pos,
            "attn_bias": attn_bias,
            "labels": torch.stack([data.y for data in batch]) 
        }

    def __call__(self, batch):
        if self.model_name == 'graphgps':
            return self.__graphgps_collate(batch)

        return self.__graphormer_collate(batch)