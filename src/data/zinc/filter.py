from rdkit import Chem

class ChemistryFilter:
    """Handles domain-specific chemistry logic using RDKit."""
    
    def __init__(self, smarts_patterns: list):
        self.patterns = []
        for smarts in smarts_patterns:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                raise ValueError(f"Invalid SMARTS pattern provided: {smarts}")
            self.patterns.append(pattern)

    def has_pattern(self, mol: Chem.Mol) -> bool:
        """Returns True if the molecule contains ANY of the SMARTS substructures."""
        if mol is None:
            return False
            
        for pattern in self.patterns:
            if mol.HasSubstructMatch(pattern):
                return True
        return False