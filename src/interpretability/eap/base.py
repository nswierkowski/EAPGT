import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable

class BaseEAP(ABC):
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        self.clean_activations: Dict[str, torch.Tensor] = {}
        self.corrupted_activations: Dict[str, torch.Tensor] = {}
        self.corrupted_gradients: Dict[str, torch.Tensor] = {}
        
        self._handles = []

    @abstractmethod
    def get_target_modules(self) -> Dict[str, nn.Module]:
        pass

    def _clean_forward_hook(self, name: str) -> Callable:
        def hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                self.clean_activations[name] = output.detach().clone()
            elif isinstance(output, tuple):
                self.clean_activations[name] = output[0].detach().clone()
        return hook

    def _corrupted_forward_hook(self, name: str) -> Callable:
        def hook(module, inputs, output):
            out_tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(out_tensor, torch.Tensor):
                self.corrupted_activations[name] = out_tensor.detach().clone()

                def grad_hook(grad):
                    self.corrupted_gradients[name] = grad.detach().clone()
                
                if out_tensor.requires_grad:
                    out_tensor.register_hook(grad_hook)
        return hook


    def _clean_message_hook(self, name: str) -> Callable:
        def hook(module, inputs, output):
            self.clean_activations[name] = output.detach().clone()
        return hook

    def _corrupted_message_hook(self, name: str) -> Callable:
        def hook(module, inputs, output):
            self.corrupted_activations[name] = output.detach().clone()
            
            def grad_hook(grad):
                self.corrupted_gradients[name] = grad.detach().clone()
            
            if output.requires_grad:
                output.register_hook(grad_hook)
        return hook

    def register_clean_hooks(self):
        self.remove_hooks()
        for name, module in self.get_target_modules().items():
            if isinstance(module, MessagePassing):
                handle = module.register_message_forward_hook(self._clean_message_hook(name))
            else:
                handle = module.register_forward_hook(self._clean_forward_hook(name))
            self._handles.append(handle)

    def register_corrupted_hooks(self):
        self.remove_hooks()
        for name, module in self.get_target_modules().items():
            if isinstance(module, MessagePassing):
                handle = module.register_message_forward_hook(self._corrupted_message_hook(name))
            else:
                handle = module.register_forward_hook(self._corrupted_forward_hook(name))
            self._handles.append(handle)

    def remove_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def clear_cache(self):
        self.clean_activations.clear()
        self.corrupted_activations.clear()
        self.corrupted_gradients.clear()

    def compute_attributions(self) -> Dict[str, torch.Tensor]:
        attributions = {}
        for name in self.clean_activations.keys():
            if name in self.corrupted_activations and name in self.corrupted_gradients:
                act_diff = self.clean_activations[name] - self.corrupted_activations[name]
                grad = self.corrupted_gradients[name]

                attributions[name] = act_diff * grad 
        return attributions
    
    def _patch_forward_hook(self, name: str, mask: torch.Tensor) -> Callable:
        def hook(module, inputs, output):
            clean_act = output[0] if isinstance(output, tuple) else output
            corrupted_act = self.corrupted_activations[name].to(clean_act.device)
            mask_device = mask.to(clean_act.device)
            
            patched_act = mask_device * clean_act + (1 - mask_device) * corrupted_act
            
            if isinstance(output, tuple):
                return (patched_act,) + output[1:]
            return patched_act
        return hook

    def _patch_message_hook(self, name: str, mask: torch.Tensor) -> Callable:
        def hook(module, inputs, output):
            corrupted_act = self.corrupted_activations[name].to(output.device)
            mask_device = mask.to(output.device)
            return mask_device * output + (1 - mask_device) * corrupted_act
        return hook

    def register_patching_hooks(self, masks: Dict[str, torch.Tensor]):
        """Attaches hooks that replace clean activations with corrupted ones based on the mask."""
        self.remove_hooks()
        for name, module in self.get_target_modules().items():
            if name not in masks:
                continue
            
            if isinstance(module, MessagePassing):
                handle = module.register_message_forward_hook(self._patch_message_hook(name, masks[name]))
            else:
                handle = module.register_forward_hook(self._patch_forward_hook(name, masks[name]))
            self._handles.append(handle)
            
    def evaluate_pair(self, clean_batch, corrupted_batch, loss_fn) -> Dict[str, torch.Tensor]:
        """
        Orchestrates the EAP forward and backward passes for a single batch pair.
        """
        self.clear_cache()
        self.model.zero_grad()
        
        self.register_clean_hooks()
        with torch.no_grad():
            self.model(clean_batch)
            
        self.register_corrupted_hooks()
        corrupted_out = self.model(corrupted_batch)

        loss = loss_fn(corrupted_out, clean_batch.y)
        loss.backward()
        
        attributions = self.compute_attributions()
        
        self.remove_hooks()
        
        return attributions