import os
import json
import torch
import gc
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score
from typing import Dict, Any, List, Optional
from src.models.wrappers.attention_mpnn import AttentionMessage
from src.models.wrappers.base import BaseMPNNWrapper
from src.interpretability.eap.evaluator import GroundTruthEvaluator

TUPLE_MODULES = (
    'GraphormerMultiheadAttention',  
    'AttentionMessage',              
    'GraphormerAttentionMessage'
)

class ThresholdOptimizer:
    def __init__(self, engine, model, val_dataloader, test_dataloader, device, tolerance: float = 0.05, prune_heads: bool = True, dataset_name: str = "ba_shapes", config: dict = None):
        self.engine = engine
        self.model = model
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.tolerance = tolerance
        self.prune_heads = prune_heads
        self.dataset_name = dataset_name
        self.config = config
        self.gt_evaluator = GroundTruthEvaluator(dataset_name, config) if config else None

    def create_masks_from_percentile(self, attributions: Dict[str, torch.Tensor], percentile: float):
        """Converts continuous component attributions to memory-efficient boolean masks on CPU."""
        all_scores = []
        for v in attributions.values():
            all_scores.append(v.detach().cpu().flatten().abs())
        
        all_scores = torch.cat(all_scores)
        threshold_val = torch.quantile(all_scores.float(), percentile / 100.0).item()
        
        masks = {}
        for k, v in attributions.items():
            masks[k] = (v.detach().abs() >= threshold_val).cpu()
        return masks

    def compute_edge_threshold_from_percentile(self, edge_attributions: List[Dict[str, torch.Tensor]], percentile: float) -> float:
        """Computes a scalar threshold from edge attributions using a memory-safe histogram."""
        max_val = 0.0
        for graph_attr in edge_attributions:
            for v in graph_attr.values():
                m = v.detach().abs().max().item()
                if m > max_val:
                    max_val = m
        if max_val == 0:
            max_val = 1.0

        num_bins = 1000000
        bin_size = max_val / num_bins
        histogram = torch.zeros(num_bins, dtype=torch.long, device='cpu')

        for graph_attr in edge_attributions:
            for v in graph_attr.values():
                scores = v.detach().abs().cpu().flatten()
                bins = (scores / bin_size).long()
                bins = torch.clamp(bins, 0, num_bins - 1)
                histogram.add_(torch.bincount(bins, minlength=num_bins))

        total_elements = histogram.sum().item()
        cumsum = torch.cumsum(histogram, dim=0)
        
        target_count = (percentile / 100.0) * total_elements
        bin_idx = torch.searchsorted(cumsum, target_count).item()
        return float(bin_idx * bin_size)

    def evaluate_patched_model(self, component_masks: Optional[Dict[str, torch.Tensor]], edge_masks: Optional[List[Dict[str, torch.Tensor]]] = None, dataloader = None, edge_threshold: Optional[float] = None, edge_attributions: Optional[List[Dict[str, torch.Tensor]]] = None) -> float:
        """Evaluates the model under strict no_grad conditions using memory-efficient dynamic hooks."""
        if dataloader is None:
            dataloader = self.val_dataloader
            
        self.model.eval()
        all_preds = []
        all_targets = []
        
        hooks = []
        current_batch_size = [0]
        current_graph_offset = [0]

        module_component_masks = {}
        head_component_masks = {}
        
        if component_masks is not None:
            for k, mask_tensor in component_masks.items():
                if '.head_' in k:
                    base_name, head_idx = k.split('.head_')
                    head_idx = int(head_idx)
                    if base_name not in head_component_masks:
                        head_component_masks[base_name] = {}
                    head_component_masks[base_name][head_idx] = mask_tensor
                else:
                    module_component_masks[k] = mask_tensor

        for target_key, info in self.engine.target_modules.items():
            module = info['module']
            module_class_name = module.__class__.__name__
            is_attn = module_class_name in ('AttentionMessage', 'GraphormerAttentionMessage')
            
            def make_hook(name=target_key, is_attention=is_attn):
                if is_attention:
                    def hook(mod, inp, out):
                        attn_weights = out[0]
                        
                        if name in head_component_masks:
                            for head_idx, mask_val in head_component_masks[name].items():
                                m = mask_val.to(attn_weights.device, dtype=attn_weights.dtype)
                                attn_weights[:, head_idx] *= m
                                
                        offset = current_graph_offset[0]
                        bsz = current_batch_size[0]
                        
                        if edge_masks is not None:
                            batch_edge_masks = edge_masks[offset : offset + bsz]
                            for i in range(min(bsz, attn_weights.size(0))):
                                if i < len(batch_edge_masks) and name in batch_edge_masks[i]:
                                    emask = batch_edge_masks[i][name].to(attn_weights.device, dtype=attn_weights.dtype)
                                    n_h, n_r, n_c = emask.shape
                                    attn_weights[i, :n_h, :n_r, :n_c] *= emask
                        elif edge_threshold is not None and edge_attributions is not None:
                            batch_attrs = edge_attributions[offset : offset + bsz]
                            for i in range(min(bsz, attn_weights.size(0))):
                                if i < len(batch_attrs) and name in batch_attrs[i]:
                                    v_attr = batch_attrs[i][name]
                                    emask = (v_attr.detach().abs() >= edge_threshold).to(attn_weights.device, dtype=attn_weights.dtype)
                                    n_h, n_r, n_c = emask.shape
                                    attn_weights[i, :n_h, :n_r, :n_c] *= emask
                                    
                        if len(out) == 2:
                            return (attn_weights, out[1])
                        else:
                            return (attn_weights, out[1], out[2])
                    return hook
                else:
                    if name in module_component_masks:
                        m = module_component_masks[name]
                        def hook(mod, inp, out):
                            if isinstance(out, torch.Tensor):
                                return out * m.to(out.device, dtype=out.dtype)
                            return out
                        return hook
                return None

            hook_fn = make_hook()
            if hook_fn is not None:
                h = module.register_forward_hook(hook_fn)
                hooks.append(h)

        graph_offset = 0
        with torch.no_grad():
            for batch in dataloader:
                def to_device(obj, dev):
                    if hasattr(obj, 'to'): return obj.to(dev)
                    if isinstance(obj, dict): return {k: v.to(dev) if hasattr(v, 'to') else v for k, v in obj.items()}
                    return obj
                
                clean_batch = to_device(batch['clean'], self.device)
                
                if isinstance(clean_batch, dict):
                    labels = batch['clean']['labels'].cpu().numpy()
                    batch_size = len(batch['clean_list'])
                else:
                    labels = clean_batch.y.cpu().numpy()
                    batch_size = clean_batch.num_graphs if hasattr(clean_batch, 'num_graphs') else 1
                
                current_batch_size[0] = batch_size
                current_graph_offset[0] = graph_offset
                
                outputs = self.model(clean_batch)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                preds = logits.argmax(dim=-1).cpu().numpy()
                
                all_preds.extend(preds)
                all_targets.extend(labels)
                graph_offset += batch_size
                
                del clean_batch, logits, outputs
                
        for h in hooks:
            h.remove()
            
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        torch.cuda.empty_cache()
        return float(f1)

    def optimize_graph_circuit(self, edge_attributions: List[Dict[str, torch.Tensor]], save_dir: str):
        """Binary search optimization for Phase 1 Edge Percentiles with streaming histograms."""
        print("--- Starting Binary Search for Graph Circuit (Edges) ---")
        baseline_f1 = self.evaluate_patched_model(None, None, self.val_dataloader)
        target_f1 = baseline_f1 - self.tolerance
        print(f"Baseline F1: {baseline_f1:.4f} | Target F1: {target_f1:.4f}")

        print("Computing high-resolution histogram of edge attributions...")
        max_val = 0.0
        for graph_attr in edge_attributions:
            for v in graph_attr.values():
                m = v.detach().abs().max().item()
                if m > max_val:
                    max_val = m
        if max_val == 0:
            max_val = 1.0

        num_bins = 1000000
        bin_size = max_val / num_bins
        histogram = torch.zeros(num_bins, dtype=torch.long, device='cpu')

        for graph_attr in edge_attributions:
            for v in graph_attr.values():
                scores = v.detach().abs().cpu().flatten()
                bins = (scores / bin_size).long()
                bins = torch.clamp(bins, 0, num_bins - 1)
                histogram.add_(torch.bincount(bins, minlength=num_bins))

        total_elements = histogram.sum().item()
        cumsum = torch.cumsum(histogram, dim=0)

        low, high = 0.0, 100.0
        best_percentile = 0.0
        best_threshold = 0.0

        while (high - low) > 0.5:
            mid = (low + high) / 2
            print(f"Testing Edge Percentile: {mid:.2f}%", end="", flush=True)
            
            target_count = (mid / 100.0) * total_elements
            bin_idx = torch.searchsorted(cumsum, target_count).item()
            threshold_val = bin_idx * bin_size
            
            val_f1 = self.evaluate_patched_model(
                component_masks=None, 
                edge_masks=None, 
                dataloader=self.val_dataloader,
                edge_threshold=threshold_val,
                edge_attributions=edge_attributions
            )
            
            if val_f1 >= target_f1:
                print(f" | Val F1: {val_f1:.4f} -> PASSED")
                best_percentile = mid
                best_threshold = threshold_val
                low = mid
            else:
                print(f" | Val F1: {val_f1:.4f} -> FAILED")
                high = mid
                
            gc.collect()
            torch.cuda.empty_cache()

        print(f"\nOptimal Graph Circuit Edge Percentile: {best_percentile:.2f}% (Threshold: {best_threshold:.6f})")
        return best_threshold, best_percentile

    def optimize(self, attributions: Dict[str, torch.Tensor], save_dir: str, node_attributions=None):
        """Binary search optimization for Phase 2 Component Percentiles."""
        print("--- Starting Binary Search for Model Circuit (Components) ---")
        baseline_f1 = self.evaluate_patched_model(None, None, self.val_dataloader)
        target_f1 = baseline_f1 - self.tolerance

        print("Computing histogram of component attributions...")
        max_val = 0.0
        for v in attributions.values():
            m = v.detach().abs().max().item()
            if m > max_val:
                max_val = m
        if max_val == 0:
            max_val = 1.0

        num_bins = 100000
        bin_size = max_val / num_bins
        histogram = torch.zeros(num_bins, dtype=torch.long, device='cpu')

        for v in attributions.values():
            scores = v.detach().abs().cpu().flatten()
            bins = (scores / bin_size).long()
            bins = torch.clamp(bins, 0, num_bins - 1)
            histogram.add_(torch.bincount(bins, minlength=num_bins))

        total_elements = histogram.sum().item()
        cumsum = torch.cumsum(histogram, dim=0)

        low, high = 0.0, 100.0
        best_percentile = 0.0
        best_component_masks = None

        while (high - low) > 0.5:
            mid = (low + high) / 2
            print(f"Testing Component Percentile: {mid:.2f}%", end="", flush=True)
            
            target_count = (mid / 100.0) * total_elements
            bin_idx = torch.searchsorted(cumsum, target_count).item()
            threshold_val = bin_idx * bin_size
            
            reconstructed_masks = {}
            for k, v in attributions.items():
                reconstructed_masks[k] = (v.detach().abs() >= threshold_val).cpu()
                
            val_f1 = self.evaluate_patched_model(reconstructed_masks, None, self.val_dataloader)
            
            if val_f1 >= target_f1:
                print(f" | Val F1: {val_f1:.4f} -> PASSED")
                best_percentile = mid
                best_component_masks = reconstructed_masks
                low = mid
            else:
                print(f" | Val F1: {val_f1:.4f} -> FAILED")
                high = mid
                
            del reconstructed_masks
            gc.collect()
            torch.cuda.empty_cache()

        print(f"\nOptimal Model Circuit Component Percentile: {best_percentile:.2f}%")
        
        if best_component_masks is not None:
            mask_path = os.path.join(save_dir, 'optimal_masks.pt')
            torch.save(best_component_masks, mask_path)
            print(f"Saved optimal component masks to {mask_path}")
            
        return best_percentile, best_component_masks

    def run_node_selection_baselines(self, save_dir: str, dataloader = None) -> dict:
        if self.gt_evaluator is None:
            raise ValueError("GroundTruthEvaluator is not initialized.")
        if dataloader is None:
            dataloader = self.test_dataloader
        return self.gt_evaluator.evaluate_node_selection_baselines(
            save_dir=save_dir, dataloader=dataloader, model=self.model, device=self.device
        )