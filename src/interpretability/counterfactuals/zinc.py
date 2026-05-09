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
            
        # 1. KEKULIZE FIRST: Clear aromatic flags to allow safe atom surgery
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            return data
            
        match_indices = []
        for pattern in self.patterns:
            if mol.HasSubstructMatch(pattern):
                match_indices = mol.GetSubstructMatch(pattern)
                break 
                
        if not match_indices:
            return data
        
        # 2. Mutate PyG Node/Edge Tensors
        for idx in match_indices:
            if corrupted.x[idx, 0] in [7, 8]: 
                corrupted.x[idx, 0] = 6       
                
                # Zero out formal charge
                if corrupted.x.shape[1] > 3:
                    corrupted.x[idx, 3] = 0

        if hasattr(corrupted, 'edge_index') and hasattr(corrupted, 'edge_attr') and corrupted.edge_attr is not None:
            for i in range(corrupted.edge_index.shape[1]):
                src, dst = corrupted.edge_index[:, i]
                if src.item() in match_indices or dst.item() in match_indices:
                    # Note: Ensure '0' is the correct integer for SINGLE bond in your specific dataset encoding!
                    # Often in ZINC, 0=Single, 1=Double, 2=Triple, 3=Aromatic.
                    corrupted.edge_attr[i, 0] = 0 
                    
        corrupted.y = torch.tensor([0], dtype=torch.long)
        
        # 3. Mutate RDKit Molecule
        rw_mol = Chem.RWMol(mol)
        for idx in match_indices:
            atom = rw_mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() in [7, 8]:
                atom.SetAtomicNum(6)
                atom.SetFormalCharge(0) 
                atom.SetIsAromatic(False)
                # Clear lingering implicit H/radical caches so RDKit recalcs them naturally
                atom.SetNumExplicitHs(0)
                atom.SetNumRadicalElectrons(0)
        
        for b in rw_mol.GetBonds():
            if b.GetBeginAtomIdx() in match_indices or b.GetEndAtomIdx() in match_indices:
                b.SetBondType(Chem.BondType.SINGLE)
                b.SetIsAromatic(False) # Ensure the bond isn't accidentally locked as aromatic
                
        # 4. Safe Sanitization
        # Update property cache before full sanitization to prevent valency conflicts
        rw_mol.UpdatePropertyCache(strict=False) 
        Chem.SanitizeMol(rw_mol)
        
        # Convert back to SMILES
        corrupted.smiles = Chem.MolToSmiles(rw_mol)
            
        return corrupted