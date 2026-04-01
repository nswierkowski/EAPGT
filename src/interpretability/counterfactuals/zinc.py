import torch
import copy
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from src.interpretability.counterfactuals.base import CounterfactualEngine

RDLogger.DisableLog('rdApp.warning')

class ZINCCounterfactual(CounterfactualEngine):
    def __init__(self, config):
        super().__init__(config)

        self.rxn_remove_no2 = AllChem.ReactionFromSmarts('[C:1][N+](=O)[O-]>>[C:1][H]')
        self.rxn_remove_nh2 = AllChem.ReactionFromSmarts('[C:1][NX3;H2]>>[C:1][H]')
        
        self.rxn_add_nh2 = AllChem.ReactionFromSmarts('[C;H1,H2,H3:1]>>[C:1]N')

    def generate(self, data):
        if not hasattr(data, 'smiles'):
            return data
            
        mol = Chem.MolFromSmiles(data.smiles)
        if mol is None: 
            return data
        
        if data.y.item() == 1:
            out = self.rxn_remove_no2.RunReactants((mol,))
            if not out: 
                out = self.rxn_remove_nh2.RunReactants((mol,))
            
            if out:
                new_mol = out[0][0]
            else:
                new_mol = mol 
        
        else:
            out = self.rxn_add_nh2.RunReactants((mol,))
            if out:
                new_mol = out[0][0] 
            else:
                new_mol = mol

            new_mol.UpdatePropertyCache()
            Chem.SanitizeMol(new_mol)
            
            new_mol = Chem.AddHs(new_mol)
            new_smiles = Chem.MolToSmiles(new_mol)
            new_y = 1 if data.y.item() == 0 else 0

        corrupted = copy.deepcopy(data)
        corrupted.smiles = new_smiles
        corrupted.y = torch.tensor([new_y], dtype=torch.long)
        
        return corrupted