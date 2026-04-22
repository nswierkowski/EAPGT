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
    """Downloads ZINC CSV, extracts graphs, preserves distribution, applies stratified splits and transforms, and saves."""
    
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
        data = self.graphs[idx]
        data.smiles = self.smiles_list[idx]
        return data

    @property
    def raw_file_names(self):
        return ['zinc_raw.csv']

    @property
    def processed_file_names(self):
        return ['zinc_no2_data.pt']

    def download(self):
        print(f"Downloading raw ZINC CSV from {self.config['dataset']['csv_url']}...")
        urllib.request.urlretrieve(self.config['dataset']['csv_url'], self.raw_paths[0])

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
        
        seed = self.config.get('seed', self.config.get('generation', {}).get('seed', 42))
        
        random.seed(seed)
        random.shuffle(pos_graphs)
        random.shuffle(neg_graphs)
        
        splits = self.config['splits']
        
        def assign_splits(graph_list):
            num_graphs = len(graph_list)
            train_end = int(splits['train'] * num_graphs)
            val_end = train_end + int(splits.get('val', 0.1) * num_graphs)
            test_end = val_end + int(splits.get('test', 0.1) * num_graphs)
            
            for i, data in enumerate(graph_list):
                if i < train_end:
                    data.split_mask = torch.tensor([0])
                elif i < val_end:
                    data.split_mask = torch.tensor([1])
                elif i < test_end:
                    data.split_mask = torch.tensor([2])
                else:
                    data.split_mask = torch.tensor([3])

        assign_splits(pos_graphs)
        assign_splits(neg_graphs)
        
        all_graphs = pos_graphs + neg_graphs
        random.shuffle(all_graphs)
        print(f"Total dataset size: {len(all_graphs)} graphs (Seed: {seed}).")

        saved_smiles = []
        processed_graphs = []
        
        for data in all_graphs:
            saved_smiles.append(data.smiles)
            del data.smiles 
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)
                
            processed_graphs.append(data)
        
        torch.save(saved_smiles, os.path.join(self.processed_dir, 'smiles.pt'))        
        torch.save(processed_graphs, self.processed_paths[0])