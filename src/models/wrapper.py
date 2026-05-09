import torch
import torch.nn as nn
from typing import List
from src.models.wrappers.res_conv_gated import WrappedResGatedGraphConv
from src.models.wrappers.attention_mpnn import WrappedAttentionMPNN
from src.models.wrappers.graphormer_multihead_attention import WrappedGraphormerMultiheadAttention
from src.models.wrappers.graph_attn_bias import WrappedGraphormerGraphAttnBias


KNOWN_MPNN_WRAPPERS = {
    'ResGatedGraphConv': WrappedResGatedGraphConv,
    'MultiheadAttention': WrappedAttentionMPNN, 
    'GraphormerMultiheadAttention': WrappedGraphormerMultiheadAttention,
    'GraphormerGraphAttnBias': WrappedGraphormerGraphAttnBias
}

def instrument_model(model: nn.Module, target_layer_names: List[str], tab: int = 0) -> nn.Module:
    """
    Recursively traverses the model and replaces target PyG layers with 
    their explicitly defined NNsight-compatible wrappers.
    
    Args:
        model: The base PyTorch model.
        target_layer_names: A list of class names (strings) from the YAML config.
    """
    target_layer_names_set = set(target_layer_names)
    
    for name, child in model.named_children():
        child_class_name = child.__class__.__name__
        
        if child_class_name in target_layer_names_set:
            
            if child_class_name not in KNOWN_MPNN_WRAPPERS:
                raise NotImplementedError(
                    f"Requested layer '{child_class_name}' is not supported. "
                    f"Please implement its wrapper and add it to KNOWN_MPNN_WRAPPERS."
                )
                
            wrapper_class = KNOWN_MPNN_WRAPPERS[child_class_name]
            
            wrapped_layer = wrapper_class(child)
            
            setattr(model, name, wrapped_layer)
            #print(f"{'  ' * tab}Wrapped layer: {name} ({child_class_name}) with {wrapper_class.__name__}")
        else:
            #print(f"{'  ' * tab}Traversing layer: {name} ({child_class_name})")
            instrument_model(child, target_layer_names, tab + 1)
            
    return model