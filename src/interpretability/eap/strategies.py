import torch.nn as nn
from typing import Dict
from .base import BaseEAP

class AbstractStrategyEAP(BaseEAP):
    """
    Provides a shared utility to normalize the patchable components 
    returned by the different GNN architectures into a flat dictionary.
    """
    def _get_flat_components(self) -> Dict[str, nn.Module]:
        raw_components = self.model.get_patchable_components()
        flat_components = {}
        
        for key, val in raw_components.items():
            if val is None:
                continue
                
            if isinstance(val, list) or isinstance(val, nn.ModuleList):
                for i, mod in enumerate(val):
                    flat_components[f"{key}_{i}"] = mod
            elif isinstance(val, dict):
                for k, v in val.items():
                    flat_components[f"{key}_{k}"] = v
            else:
                flat_components[key] = val
                
        return flat_components


class ClassicEAP(AbstractStrategyEAP):
    """
    NLP-style EAP. 
    Strictly targets Transformer components like Attention and MLPs.
    """
    def get_target_modules(self) -> Dict[str, nn.Module]:
        all_components = self._get_flat_components()

        return {k: v for k, v in all_components.items() if 'classic_' in k}


class MinarEAP(AbstractStrategyEAP):
    """
    Graph-style EAP.
    Strictly targets Message Passing layers and topological encodings.
    """
    def get_target_modules(self) -> Dict[str, nn.Module]:
        all_components = self._get_flat_components()

        return {k: v for k, v in all_components.items() if 'minar_' in k}


class HybridEAP(AbstractStrategyEAP):
    """
    Full Architecture EAP.
    Targets all components returned by the model.
    """
    def get_target_modules(self) -> Dict[str, nn.Module]:
        return self._get_flat_components()