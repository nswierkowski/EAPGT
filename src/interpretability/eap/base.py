import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable
from nnsight import NNsight
from src.models.wrappers.attention_mpnn import AttentionMessage

tuple_modules = (
    'GraphormerMultiheadAttention',  
    'AttentionMessage',              
    'GraphormerAttentionMessage'
)

class BaseEAP(ABC):
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        self.tracer = NNsight(self.model)
        self.graph_states: Dict[str, torch.Tensor] = {}
        
        raw_target_modules = self.get_target_modules()
        self.target_modules = {}
        id_to_path = {id(m): name for name, m in self.model.named_modules()}
        
        for key, raw_mod in raw_target_modules.items():
            path = id_to_path.get(id(raw_mod))
            if path is not None:
                proxy = self.tracer
                for part in path.split('.'):
                    if part: 
                        proxy = getattr(proxy, part)
                self.target_modules[key] = {
                    'proxy': proxy,
                    'path': path,
                    'module': raw_mod
                }
            else:
                print(f"WARNING: Target module {key} was not found in the model tree!")

    @abstractmethod
    def get_target_modules(self) -> Dict[str, nn.Module]:
        pass
    
    def _get_proxy(self, path: str):
        """Dynamically resolves the proxy inside the current trace context."""
        proxy = self.tracer
        for part in path.split('.'):
            if part:
                proxy = getattr(proxy, part)
        return proxy
    
    def _extract_tensor(self, proxy: Any, module: nn.Module) -> Any:
        out = proxy.output
        
        if module.__class__.__name__ == 'EncodingInterceptor':
            return out
        
        if isinstance(module, nn.MultiheadAttention) or module.__class__.__name__ in tuple_modules:
            return out[0]
        
        return out

    def evaluate_pair(self, clean_batch: Any, corrupted_batch: Any, loss_fn: Callable) -> Dict[str, Dict[str, torch.Tensor]]:
        target_modules = self.target_modules
        
        corrupted_acts = {}
        clean_acts = {}
            
        with self.tracer.trace(corrupted_batch):
            for name, data in target_modules.items():
                proxy = self._get_proxy(data['path'])
                tensor = self._extract_tensor(proxy, data['module'])
                corrupted_acts[name] = tensor.save()

        # 2. Forward & Backward Trace: Clean Batch
        with self.tracer.trace(clean_batch):
            for name, data in target_modules.items():
                proxy = self._get_proxy(data['path'])
                tensor = self._extract_tensor(proxy, data['module'])
                
                tensor.retain_grad() 
                
                saved_proxy = tensor.save()
                self.graph_states[name] = saved_proxy
                clean_acts[name] = saved_proxy
                
                # Notice we removed clean_grads[name] = tensor.grad.save() here!

            logits = self.tracer.output
            if hasattr(logits, 'logits'): 
                logits = logits.logits
                
            labels = clean_batch.y if hasattr(clean_batch, 'y') else clean_batch['labels']
            labels = labels.view(-1).long()

            loss = loss_fn(logits, labels)
            loss.backward()
            
        macro_attributions = {}
        micro_attributions = {}
        
        # 3. Calculate Macro and Micro EAP Scores
        for name in target_modules.keys():
            act_clean = clean_acts[name]
            act_corr = corrupted_acts[name]
            
            grad_clean = act_clean.grad
            
            if grad_clean is None:
                raise RuntimeError(f"Gradient for '{name}' is None!")

            if act_clean.shape != act_corr.shape or act_clean.shape != grad_clean.shape:
                raise RuntimeError(f"Shape mismatch in {name}: Clean {act_clean.shape}, Corr {act_corr.shape}, Grad {grad_clean.shape}")

            act_clean_detached = act_clean.detach()
            act_corr_detached = act_corr.detach()
            grad_clean_detached = grad_clean.detach()

            act_diff = act_clean_detached - act_corr_detached
            eap_tensor = act_diff * grad_clean_detached
            
            # --- MACRO-EAP ---
            macro_score = eap_tensor.sum().abs()
            macro_attributions[name] = macro_score.cpu().item()
            
            # --- MICRO-EAP ---
            if name.endswith('.M') or 'message' in name.lower():
                if "attn" in name.lower() or eap_tensor.dim() == 4:
                    micro_score = eap_tensor.abs()
                else:
                    micro_score = eap_tensor.sum(dim=-1).abs()
                    
                micro_attributions[name] = micro_score.cpu()

        del clean_acts, corrupted_acts
        self.graph_states.clear()         
        torch.cuda.empty_cache()

        return {
            'macro': macro_attributions,
            'micro': micro_attributions
        }
        
    def evaluate_pair_eap_ig(self, clean_batch: Any, corrupted_batch: Any, loss_fn: Callable, steps: int = 10) -> Dict[str, torch.Tensor]:
        """
        Calculates Edge Attribution Patching using Integrated Gradients (EAP-IG).
        Solves gradient saturation for confident models by interpolating between states.
        """
        target_modules = self.target_modules
        
        corrupted_acts = {}
        clean_acts = {}
        
        # 1. Base passes to extract endpoints (no gradients required here)
        with self.tracer.trace(corrupted_batch):
            for name, data in target_modules.items():
                proxy = self._get_proxy(data['path'])
                real_module = data['module']
                out = proxy.output
                if isinstance(real_module, nn.MultiheadAttention) or real_module.__class__.__name__ in tuple_modules:
                    out = out[0]
                corrupted_acts[name] = out.save()

        with self.tracer.trace(clean_batch):
            for name, data in target_modules.items():
                proxy = self._get_proxy(data['path'])
                real_module = data['module']
                out = proxy.output
                if isinstance(real_module, nn.MultiheadAttention) or real_module.__class__.__name__ in tuple_modules:
                    out = out[0]
                
                self.graph_states[name] = out.save()
                clean_acts[name] = out.save()

        # Detach endpoints to use as constants during interpolation
        clean_vals = {name: act.detach() for name, act in clean_acts.items()}
        corr_vals = {name: act.detach() for name, act in corrupted_acts.items()}
        
        # Initialize gradient accumulators
        accumulated_grads = {name: torch.zeros_like(clean_vals[name]) for name in target_modules}

        # 2. Integrated Gradients Interpolation passes
        for step in range(1, steps + 1):
            alpha = step / steps
            intervened_acts = {}
            with self.tracer.trace(clean_batch):
                
                for name, data in target_modules.items():
                    proxy = self._get_proxy(data['path'])
                    real_module = data['module']
                    out = proxy.output
                    is_tuple = isinstance(real_module, nn.MultiheadAttention) or 'Attention' in real_module.__class__.__name__
                    
                    # Compute interpolated activation
                    c_val = clean_vals[name]
                    cr_val = corr_vals[name]
                    interpolated = cr_val + alpha * (c_val - cr_val)
                    
                    interpolated.requires_grad_(True)
                    # Intervene in the NNsight proxy graph
                    if is_tuple:
                        proxy.output = (interpolated, out[1])
                    else:
                        proxy.output = interpolated
                    
                    # Retain grad on the intervened output specifically
                    act_to_grad = proxy.output[0] if is_tuple else proxy.output
                    act_to_grad.retain_grad()
                    intervened_acts[name] = act_to_grad.save()
                    
                labels = clean_batch.y if hasattr(clean_batch, 'y') else clean_batch['labels']
                logits = self.tracer.output.logits if hasattr(self.tracer.output, 'logits') else self.tracer.output
                
                # We still use your loss_fn, whether it's CrossEntropy or Logit Diff
                loss = loss_fn(logits, labels)
                loss.backward()
                
            # Accumulate gradients from this interpolation step
            for name in target_modules:
                grad = intervened_acts[name].grad
                if grad is None:
                    raise RuntimeError(f"Gradient for '{name}' at step {step} is None!")
                accumulated_grads[name] += grad.detach()

        # 3. Compute final IG attributions
        attributions = {}
        for name in target_modules.keys():
            act_diff = clean_vals[name] - corr_vals[name]
            avg_grad = accumulated_grads[name] / steps
            
            # EAP-IG Score formulation
            eap_score = (act_diff * avg_grad).sum(dim=0).abs()
            
            if eap_score.dim() == 3: 
                num_heads = eap_score.shape[0]
                for h in range(num_heads):
                    attributions[f"{name}.head_{h}"] = eap_score[h]
            else:
                attributions[name] = eap_score

        del clean_acts, corrupted_acts, clean_vals, corr_vals, accumulated_grads
        torch.cuda.empty_cache()

        return attributions