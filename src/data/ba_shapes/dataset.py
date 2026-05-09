import os
import torch
from typing import Dict, Optional, Callable, List
from torch_geometric.data import InMemoryDataset, Data
from src.data.ba_shapes.generator import BAShapesGenerator

class BAShapesDataset(InMemoryDataset):
    def __init__(self, root: str, config: dict, pre_transform: Optional[Callable] = None):
        self.config = config
        self.num_samples = config['dataset']['num_samples']# .get('num_samples', 1000)
        
        if 'generation' in config:
            self.generator = BAShapesGenerator(
                num_base_nodes=config['generation'].get('num_base_nodes', 15),
                m_edges=config['generation'].get('m_edges', 1)
            )
        super().__init__(root, transform=None, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return ['ba_shapes_graph_level.pt']

    def process(self):
        data_list = []
        feature_dim = self.config.get('feature_dim', 10)

        for i in range(self.num_samples):
            has_motif = (i < self.num_samples // 2)
            nx_graph, label = self.generator.generate_sample(has_motif)
            
            edges = list(nx_graph.edges())
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            
            x = torch.ones((nx_graph.number_of_nodes(), feature_dim), dtype=torch.float)
            y = torch.tensor([label], dtype=torch.long)
            
            data = Data(x=x, edge_index=edge_index, y=y)
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)
                
            data_list.append(data)
        
        num_graphs = len(data_list)
        seed = self.config.get('experiment', {}).get('random_state', 42)
        g = torch.Generator()
        g.manual_seed(seed)
        
        indices = torch.randperm(num_graphs, generator=g)
        split_ratios = self.config.get('splits', {'train': 0.7, 'val': 0.1, 'test': 0.2, 'eap': 0.0})
        
        train_end = int(split_ratios['train'] * num_graphs)
        val_end = train_end + int(split_ratios['val'] * num_graphs)
        test_end = val_end + int(split_ratios['test'] * num_graphs)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:test_end]
        eap_indices = indices[test_end:]
        
        for idx in train_indices:
            data_list[idx].split_mask = torch.tensor(0)
        for idx in val_indices:
            data_list[idx].split_mask = torch.tensor(1)
        for idx in test_indices:
            data_list[idx].split_mask = torch.tensor(2)
        for idx in eap_indices:
            data_list[idx].split_mask = torch.tensor(3)
        
        torch.save(self.collate(data_list), self.processed_paths[0])


    def get_split_indices(self) -> Dict[str, torch.Tensor]:
        """Returns deterministic indices for Graph-level splits (Train/Val/Test/EAP)."""
        num_graphs = len(self)
        
        seed = self.config.get('experiment', {}).get('random_state', 42)
        g = torch.Generator()
        g.manual_seed(seed)
        
        indices = torch.randperm(num_graphs, generator=g)
        
        splits = self.config.get('splits', {'train': 0.7, 'val': 0.1, 'test': 0.2, 'eap': 0.0})
        
        train_end = int(splits['train'] * num_graphs)
        val_end = train_end + int(splits['val'] * num_graphs)
        test_end = val_end + int(splits['test'] * num_graphs)
        
        return {
            'train': indices[:train_end],
            'val': indices[train_end:val_end],
            'test': indices[val_end:test_end],
            'eap': indices[test_end:]
        }