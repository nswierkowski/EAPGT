import torch
from torch_geometric.transforms import BaseTransform, AddRandomWalkPE, AddLaplacianEigenvectorPE

class GraphGPSTransform(BaseTransform):
    def __init__(self, model_config: dict):
        self.pe_type = model_config.get('pe_type', 'rwpe').lower()
        self.walk_length = model_config.get('walk_length', 20)
        
        if self.pe_type == 'rwpe':
            self.transform = AddRandomWalkPE(
                walk_length=self.walk_length, 
                attr_name='pe' 
            )
        elif self.pe_type == 'lappe':
            pe_dim = model_config.get('pe_dim', 8)
            self.transform = AddLaplacianEigenvectorPE(
                k=pe_dim, 
                attr_name='pe',
                is_undirected=True
            )
        else:
            raise ValueError(f"Unsupported PE type for GraphGPS: {self.pe_type}")

    def forward(self, data):
        data = self.transform(data)
        
        if hasattr(data, 'pe') and data.pe is not None:
            data.pe = data.pe.to(torch.float32)
            
        return data