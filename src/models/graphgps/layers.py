import torch
import torch.nn as nn

class FeatureEncoder(nn.Module):
    """Encodes nodes and edges conditionally based on the dataset."""
    def __init__(self, config):
        super().__init__()
        dataset_name = config.get('dataset', {}).get('name', 'unknown').lower()
        model_config = config['model']
        
        hidden_dim = model_config['hidden_dim']
        
        if dataset_name in ['ba_shapes']:
            
            input_dim = model_config.get('input_dim', 10)
            self.node_encoder = nn.Linear(input_dim, hidden_dim)
            self.is_discrete_edges = False
        else:
            self.node_encoder = nn.Embedding(100, hidden_dim) 
            self.is_discrete_edges = True
            
        if self.is_discrete_edges:
            self.edge_encoder = nn.Embedding(10, hidden_dim)
        else:
            self.edge_encoder = None

        pe_dim = model_config.get('pe_dim', 16)
        self.pe_encoder = nn.Linear(pe_dim, hidden_dim)

    def forward(self, x, edge_attr, pe, edge_index):

        if not self.is_discrete_edges:
            x = x.to(torch.float32)
            x_emb = self.node_encoder(x)
            e_emb = torch.zeros(edge_index.size(1), x_emb.size(1), device=x.device)
        else:
            x = x.to(torch.long)
            x_emb = self.node_encoder(x.squeeze(-1) if x.dim() > 1 else x)
            
            if edge_attr is not None:
                edge_attr = edge_attr.to(torch.long)
                e_emb = self.edge_encoder(edge_attr.squeeze(-1) if edge_attr.dim() > 1 else edge_attr)
            else:
                e_emb = torch.zeros(edge_index.size(1), x_emb.size(1), device=x.device)

        if pe is not None:
            pe = pe.to(torch.float32)
            x_emb = x_emb + self.pe_encoder(pe)

        return x_emb, e_emb