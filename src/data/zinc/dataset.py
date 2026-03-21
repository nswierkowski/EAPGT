import os
import torch
import pandas as pd
import urllib.request
from typing import Optional, Callable
from torch_geometric.data import InMemoryDataset
from rdkit import Chem
import random

from src.data.zinc.filter import ChemistryFilter
from src.data.zinc.converter import MolToGraphConverter

class ZINCNO2Dataset(InMemoryDataset):
    """Downloads ZINC CSV, extracts graphs, balances labels, applies transforms, and saves."""
    
    def __init__(self, root: str, config: dict, pre_transform: Optional[Callable] = None):
        self.config = config
        self.chem_filter = ChemistryFilter(config['generation']['smarts_patterns'])
        self.converter = MolToGraphConverter()
        
        super().__init__(root, transform=None, pre_transform=pre_transform)
        
        self.graphs = torch.load(self.processed_paths[0], weights_only=False)
        self.smiles_list = torch.load(os.path.join(self.processed_dir, 'smiles.pt'), weights_only=False)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

    @property
    def raw_file_names(self):
        return ['zinc_raw.csv']

    @property
    def processed_file_names(self):
        return ['zinc_no2_data.pt']

    def download(self):
        print(f"Downloading raw ZINC CSV from {self.config['csv_url']}...")
        urllib.request.urlretrieve(self.config['csv_url'], self.raw_paths[0])

    def process(self):
        print("Parsing SMILES and extracting graphs...")
        df = pd.read_csv(self.raw_paths[0])
        smiles_list = df['smiles'].dropna().tolist()
        
        limit = self.config['generation'].get('max_molecules', len(smiles_list))
        
        pos_graphs = []
        neg_graphs = []
        
        for smiles in smiles_list[:limit]:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: 
                continue
            
            mol = Chem.AddHs(mol)
                
            has_no2 = self.chem_filter.has_pattern(mol)
            label = 1 if has_no2 else 0
            
            data = self.converter.convert(mol, label)
            data.smiles = smiles
                        
            if has_no2:
                pos_graphs.append(data)
            else:
                neg_graphs.append(data)

        print(f"Found {len(pos_graphs)} positive and {len(neg_graphs)} negative samples.")
        min_len = min(len(pos_graphs), len(neg_graphs))
        
        random.seed(42)
        random.shuffle(neg_graphs)
        
        balanced_graphs = pos_graphs[:min_len] + neg_graphs[:min_len]
        random.shuffle(balanced_graphs)
        print(f"Balanced dataset size: {len(balanced_graphs)} graphs.")

        splits = self.config['splits']
        num_graphs = len(balanced_graphs)
        train_end = int(splits['train'] * num_graphs)
        val_end = train_end + int(splits['val'] * num_graphs)
        test_end = val_end + int(splits['test'] * num_graphs)

        saved_smiles = []
        processed_graphs = []
        
        for i, data in enumerate(balanced_graphs):
            saved_smiles.append(data.smiles)
            del data.smiles 
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            if i < train_end:
                data.split_mask = torch.tensor([0]) 
            elif i < val_end:
                data.split_mask = torch.tensor([1]) 
            elif i < test_end:
                data.split_mask = torch.tensor([2]) 
            else:
                data.split_mask = torch.tensor([3]) 
                
            processed_graphs.append(data)
        
        torch.save(saved_smiles, os.path.join(self.processed_dir, 'smiles.pt'))        
        torch.save(processed_graphs, self.processed_paths[0])
    