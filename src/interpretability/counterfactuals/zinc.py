import torch
import copy
from rdkit import Chem
from src.interpretability.counterfactuals.base import CounterfactualEngine

class ZINCCounterfactual(CounterfactualEngine):
    def __init__(self, config):
        super().__init__(config)
        smarts = config.get('generation', {}).get('smarts_patterns', ['[N+](=O)[O-]', '[NX3;H2]'])
        self.patterns = [Chem.MolFromSmarts(s) for s in smarts]

    def generate(self, data):
        if data.y.item() == 0:
            return data
            
        corrupted = copy.deepcopy(data)
        
        mol = Chem.MolFromSmiles(corrupted.smiles)
        if mol is None:
            return data
        mol = Chem.AddHs(mol) 
        match_indices = []
        for pattern in self.patterns:
            if mol.HasSubstructMatch(pattern):
                match_indices = mol.GetSubstructMatch(pattern)
                break 
                
        if not match_indices:
            return data
        
        for idx in match_indices:
            if corrupted.x[idx, 0] in [7, 8]:
                corrupted.x[idx, 0] = 6
                
        corrupted.y = torch.tensor([0], dtype=torch.long)
        
        rw_mol = Chem.RWMol(mol)
        for idx in match_indices:
            atom = rw_mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() in [7, 8]:
                atom.SetAtomicNum(6)
                atom.SetFormalCharge(0) 
                atom.SetIsAromatic(False)
                
            Chem.SanitizeMol(rw_mol)
            
        corrupted.smiles = Chem.MolToSmiles(rw_mol)
        
        return corrupted