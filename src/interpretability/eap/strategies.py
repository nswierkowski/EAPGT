from src.models.wrappers.attention_mpnn import WrappedAttentionMPNN
from src.models.wrappers.base import BaseMPNNWrapper
import torch.nn as nn
from collections import OrderedDict
from typing import Dict
from .base import BaseEAP


class MacroMinarEAP(BaseEAP):
    def get_target_modules(self) -> dict[str, nn.Module]:
        target_modules = OrderedDict()
        
        # --- NEW: Blacklist conditionally bypassed encoders ---
        skip_prefixes = [
            'encoder.pe_encoder', 
            'pe_encoder',
            'encoder.node_encoder', # Optional: skip node encoder for speed
            'encoder.edge_encoder'  # Optional: skip edge encoder for speed
        ]
        
        valid_leaf_types = (
            nn.Linear
        ) 

        for name, module in self.model.named_modules():
            if name == "": 
                continue 
            
            if any(name.startswith(p) for p in skip_prefixes):
                continue

            if isinstance(module, BaseMPNNWrapper) or isinstance(module, WrappedAttentionMPNN):
                target_modules[f"{name}.M"] = module.message_module
                target_modules[f"{name}.A"] = module.aggregate_module
                target_modules[f"{name}.U"] = module.update_module
                skip_prefixes.append(f"{name}.")
                
            elif isinstance(module, nn.MultiheadAttention) or module.__class__.__name__ == 'GraphormerMultiheadAttention':
                target_modules[name] = module
                skip_prefixes.append(f"{name}.") 
                
            elif module.__class__.__name__ == 'WrappedGraphormerGraphAttnBias':
                target_modules[f"{name}.spatial_encoding"] = module.spatial_tracker
                target_modules[f"{name}.edge_encoding"] = module.edge_tracker
                target_modules[f"{name}.combined_bias"] = module.combined_tracker
                
                skip_prefixes.append(f"{name}.")
                
            elif isinstance(module, valid_leaf_types):
                target_modules[name] = module
                
        if not target_modules:
            print("WARNING: No target modules found.")
            
        return target_modules