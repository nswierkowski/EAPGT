import torch
from torch_geometric.data import Data
from rdkit import Chem

class MolToGraphConverter:
    """Converts RDKit Mol objects into PyTorch Geometric Data objects for Graphormer."""
    
    def convert(self, mol: Chem.Mol, label: int) -> Data:
        x = torch.tensor([[atom.GetAtomicNum()] for atom in mol.GetAtoms()], dtype=torch.long)
        
        edge_indices = []
        edge_attrs = []
        
        bond_type_map = {
            Chem.rdchem.BondType.SINGLE: 1,
            Chem.rdchem.BondType.DOUBLE: 2,
            Chem.rdchem.BondType.TRIPLE: 3,
            Chem.rdchem.BondType.AROMATIC: 4
        }
        
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]
            
            b_type = bond_type_map.get(bond.GetBondType(), 1)
            edge_attrs += [[b_type], [b_type]]

        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.long)

        y = torch.tensor([label], dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)