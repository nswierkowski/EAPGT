import os
import json
import torch
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score
from typing import Dict, Any, List, Optional
from src.models.wrappers.attention_mpnn import AttentionMessage
from src.models.wrappers.base import BaseMPNNWrapper
from src.interpretability.eap.evaluator import GroundTruthEvaluator
import pandas as pd

# Centralize module tuple checks
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
        """
        Converts continuous attributions to binary masks based on a percentile.
        Automatically recombines .head_X keys back into a single tensor for patching.
        """
        all_scores = torch.cat([v.flatten() for v in attributions.values()])
        threshold_val = torch.quantile(all_scores.float(), percentile / 100.0).item()
        
        flat_masks = {}
        for name, score_tensor in attributions.items():
            flat_masks[name] = (score_tensor >= threshold_val).float()
            
        # --- ENFORCE MPNN DEPENDENCIES (M -> A -> U) ---
        # Rule 1: If M is cut, A and U must be cut.
        # Rule 2: If A or U cannot be cut, M should not be cut.
        mpnn_prefixes = set()
        for name in flat_masks.keys():
            for part in ['.M', '.A', '.U']:
                if part in name:
                    mpnn_prefixes.add(name.split(part)[0])
        
        for prefix in mpnn_prefixes:
            m_keys = [k for k in flat_masks.keys() if k.startswith(prefix + ".M")]
            a_keys = [k for k in flat_masks.keys() if k.startswith(prefix + ".A")]
            u_keys = [k for k in flat_masks.keys() if k.startswith(prefix + ".U")]
            
            def get_head_map(keys):
                hmap = {}
                for k in keys:
                    if '.head_' in k:
                        h_idx = k.split('.head_')[-1]
                        hmap[h_idx] = k
                    else:
                        hmap['global'] = k
                return hmap
            
            m_map = get_head_map(m_keys)
            a_map = get_head_map(a_keys)
            u_map = get_head_map(u_keys)
            
            all_heads = set(m_map.keys()) | set(a_map.keys()) | set(u_map.keys())
            
            for h in all_heads:
                mk = m_map.get(h, m_map.get('global'))
                ak = a_map.get(h, a_map.get('global'))
                uk = u_map.get(h, u_map.get('global'))
                
                # Tie M, A, and U components together: if any is kept, all are kept.
                found_keys = [k for k in [mk, ak, uk] if k and k in flat_masks]
                if found_keys:
                    combined_mask = torch.max(torch.stack([flat_masks[k] for k in found_keys]), dim=0)[0]
                    for k in found_keys:
                        flat_masks[k] = combined_mask
            
        reconstructed_masks = {}
        head_groups = {}
        
        for name, mask in flat_masks.items():
            if self.prune_heads and ".head_" in name:
                parent_name = name.split(".head_")[0]
                head_idx = int(name.split(".head_")[1])
                if parent_name not in head_groups:
                    head_groups[parent_name] = {}
                head_groups[parent_name][head_idx] = mask
            else:
                reconstructed_masks[name] = mask
                
        for parent_name, heads in head_groups.items():
            num_heads = max(heads.keys()) + 1
            stacked_mask = torch.stack([heads[i] for i in range(num_heads)], dim=0)
            reconstructed_masks[parent_name] = stacked_mask
            
        return flat_masks, reconstructed_masks, threshold_val

    def create_edge_masks_from_percentile(self, edge_attributions: List[Dict[str, torch.Tensor]], percentile: float):
        """
        Converts edge attributions to binary masks based on a percentile.
        edge_attributions is a list of dicts (per graph).
        """
        all_scores = []
        for graph_attr in edge_attributions:
            for score in graph_attr.values():
                all_scores.append(score.flatten())
        
        if not all_scores:
            return [], 0.0
            
        all_scores = torch.cat(all_scores)
        
        # --- PREVENT MEMORY OVERFLOW / QUANTILE LIMITS ---
        print(f"Number of attributions: {all_scores.numel()}")
        max_sample_size = 5_000_000
        if all_scores.numel() > max_sample_size:
            indices = torch.randint(0, all_scores.numel(), (max_sample_size,))
            sample_scores = all_scores[indices]
            threshold_val = torch.quantile(sample_scores.float(), percentile / 100.0).item()
        else:
            threshold_val = torch.quantile(all_scores.float(), percentile / 100.0).item()
        
        edge_masks = []
        for graph_attr in edge_attributions:
            graph_mask = {}
            for name, score in graph_attr.items():
                graph_mask[name] = (score >= threshold_val).float()
            edge_masks.append(graph_mask)
            
        return edge_masks, threshold_val

    def evaluate_baseline_pure(self, dataloader=None) -> float:
        """
        Strict Trainer-style evaluation. No NNsight, no complex dictionary unpacking.
        Isolates standard PyTorch/PyG forward passes from patching frameworks.
        """
        from sklearn.metrics import precision_recall_fscore_support

        if dataloader is None:
            dataloader = self.val_dataloader

        def _process_batch(batch):
            if hasattr(batch, 'to'): 
                batch = batch.to(self.device)
                labels = batch.y
            elif isinstance(batch, dict):
                batch = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in batch.items()}
                labels = batch.get('y', batch.get('labels'))
            else:
                raise TypeError("Unrecognized batch format from collator.")
                
            if labels.dim() > 1 and labels.size(-1) == 1:
                labels = labels.squeeze(-1)
                
            return batch, labels

        def _evaluate(loader):
            self.model.eval()
            total_loss = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in tqdm(loader, desc="Pure Trainer Eval", leave=False):
                    clean_data = batch['clean']
                    batch_data, labels = _process_batch(clean_data)
                    
                    outputs = self.model(batch_data)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    
                    loss = torch.nn.CrossEntropyLoss()(logits, labels)
                    total_loss += loss.item() * labels.size(0)
                    preds = torch.argmax(logits, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
            num_samples = max(len(all_labels), 1)
            avg_loss = total_loss / num_samples
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='macro', zero_division=0
            )
            
            return avg_loss, 0.0, precision, recall, f1
        
        return _evaluate(dataloader)[-1]

    def evaluate_patched_model(self, component_masks: Dict[str, torch.Tensor] = None, edge_masks: List[Dict[str, torch.Tensor]] = None, dataloader=None) -> float:
        """Runs F1 evaluation using physical pruning for edges and NNsight causal patching for components."""
        if dataloader is None:
            dataloader = self.val_dataloader

        self.model.eval()
        all_preds = []
        all_targets = []
        graph_offset = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating F1", leave=False):
                # Setup Batch Data
                clean_data = batch['clean'] if isinstance(batch, dict) and 'clean' in batch else batch
                if hasattr(clean_data, 'to'):
                    clean_batch = clean_data.to(self.device)
                elif isinstance(clean_data, dict):
                    clean_batch = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in clean_data.items()}
                else:
                    clean_batch = clean_data
                
                labels = clean_batch.y if hasattr(clean_batch, 'y') else clean_batch['labels']
                if labels.dim() > 1 and labels.size(-1) == 1:
                    labels = labels.squeeze(-1)
                
                corrupted_batch = None
                if isinstance(batch, dict) and 'corrupted' in batch:
                    corr_data = batch['corrupted']
                    if hasattr(corr_data, 'to'):
                        corrupted_batch = corr_data.to(self.device)
                    elif isinstance(corr_data, dict):
                        corrupted_batch = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in corr_data.items()}
                    else:
                        corrupted_batch = corr_data

                if isinstance(batch, dict) and 'clean_list' in batch:
                    batch_size = len(batch['clean_list'])
                elif hasattr(clean_batch, 'num_graphs'):
                    batch_size = clean_batch.num_graphs
                elif isinstance(clean_batch, dict) and 'attn_bias' in clean_batch:
                    batch_size = clean_batch['attn_bias'].shape[0]
                else:
                    batch_size = labels.size(0)

                # --- 1. APPLY GRAPH TOPOLOGY MASKS (PHYSICAL PRUNING) ---
                if edge_masks is not None:
                    batch_edge_masks = edge_masks[graph_offset : graph_offset + batch_size]
                    
                    if hasattr(clean_batch, 'edge_index') and clean_batch.edge_index is not None:
                        bool_mask = torch.ones(clean_batch.edge_index.shape[1], dtype=torch.bool, device=self.device)
                        batch_vec = clean_batch.batch if hasattr(clean_batch, 'batch') else None
                        ptr = clean_batch.ptr if hasattr(clean_batch, 'ptr') else None
                        
                        if batch_vec is not None:
                            edge_batch_idx = batch_vec[clean_batch.edge_index[0]]
                        else:
                            edge_batch_idx = torch.zeros(clean_batch.edge_index.shape[1], dtype=torch.long, device=self.device)
                            
                        for i, g_mask_dict in enumerate(batch_edge_masks):
                            graph_edge_mask = (edge_batch_idx == i)
                            num_e = graph_edge_mask.sum().item()
                            if num_e == 0:
                                continue
                                
                            g_mask = list(g_mask_dict.values())[0] if isinstance(g_mask_dict, dict) else g_mask_dict
                            
                            if g_mask.dim() == 3:
                                g_mask = g_mask.any(dim=0) 
                            elif g_mask.dim() == 2:
                                if g_mask.shape[0] != g_mask.shape[1]:
                                    g_mask = g_mask.any(dim=0) if g_mask.shape[0] < g_mask.shape[1] else g_mask.any(dim=1)
                                    
                            g_mask_flat = g_mask.view(-1)
                            
                            if g_mask_flat.shape[0] == num_e:
                                bool_mask[graph_edge_mask] = g_mask_flat.bool()
                            elif g_mask.dim() == 2 and ptr is not None:
                                e_idx = clean_batch.edge_index[:, graph_edge_mask]
                                u_local = e_idx[0] - ptr[i]
                                v_local = e_idx[1] - ptr[i]
                                
                                valid = (u_local < g_mask.shape[0]) & (v_local < g_mask.shape[1])
                                local_bool = torch.ones(num_e, dtype=torch.bool, device=self.device)
                                local_bool[valid] = g_mask[u_local[valid], v_local[valid]].bool()
                                bool_mask[graph_edge_mask] = local_bool
                            else:
                                raise ValueError(f"Graph {i} Shape Mismatch.")
                                
                        clean_batch.edge_index = clean_batch.edge_index[:, bool_mask]
                        if hasattr(clean_batch, 'edge_attr') and clean_batch.edge_attr is not None:
                            clean_batch.edge_attr = clean_batch.edge_attr[bool_mask]
                            
                        if corrupted_batch is not None and hasattr(corrupted_batch, 'edge_index') and corrupted_batch.edge_index is not None:
                            corrupted_batch.edge_index = corrupted_batch.edge_index[:, bool_mask]
                            if hasattr(corrupted_batch, 'edge_attr') and corrupted_batch.edge_attr is not None:
                                corrupted_batch.edge_attr = corrupted_batch.edge_attr[bool_mask]
                            
                    elif isinstance(clean_batch, dict) and 'attn_bias' in clean_batch:
                        first_dict = batch_edge_masks[0]
                        if isinstance(first_dict, dict):
                            first_k = list(first_dict.keys())[0]
                            masks = []
                            for m in batch_edge_masks:
                                raw_m = m[first_k]
                                if raw_m.dim() == 1:
                                    n_nodes = int(raw_m.shape[0] ** 0.5)
                                    if n_nodes * n_nodes == raw_m.shape[0]:
                                        raw_m = raw_m.view(n_nodes, n_nodes)
                                masks.append(raw_m)
                                
                            stacked_mask = torch.stack(masks).to(self.device)
                            
                            if stacked_mask.dim() == 4:
                                stacked_mask = stacked_mask.any(dim=1)
                                
                            while stacked_mask.dim() < clean_batch['attn_bias'].dim():
                                stacked_mask = stacked_mask.unsqueeze(1)
                                
                            clean_batch['attn_bias'] = clean_batch['attn_bias'] * stacked_mask.float()
                            if corrupted_batch is not None and isinstance(corrupted_batch, dict) and 'attn_bias' in corrupted_batch:
                                corrupted_batch['attn_bias'] = corrupted_batch['attn_bias'] * stacked_mask.float()
                        else:
                            stacked_mask = torch.stack(batch_edge_masks).to(self.device)
                            
                            if stacked_mask.dim() == 4:
                                stacked_mask = stacked_mask.any(dim=1)
                                
                            clean_batch['attn_bias'] = clean_batch['attn_bias'] * stacked_mask.float()
                            if corrupted_batch is not None and isinstance(corrupted_batch, dict) and 'attn_bias' in corrupted_batch:
                                corrupted_batch['attn_bias'] = corrupted_batch['attn_bias'] * stacked_mask.float()

                # --- 2. EVALUATE MODEL (FAST PASS OR NNSIGHT PATCHING) ---
                if component_masks is None:
                    outputs = self.model(clean_batch)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    preds = logits.argmax(dim=-1).detach()
                    all_preds.append(preds.cpu())
                    all_targets.append(labels.cpu())
                else:
                    if corrupted_batch is None:
                        corr_data = batch['corrupted']
                        if hasattr(corr_data, 'to'):
                            corrupted_batch = corr_data.to(self.device)
                        elif isinstance(corr_data, dict):
                            corrupted_batch = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in corr_data.items()}
                        else:
                            corrupted_batch = corr_data
                    
                    corrupted_acts = {}
                    
                    with self.engine.tracer.trace(corrupted_batch):
                        for name, data in self.engine.target_modules.items():
                            proxy = data['proxy']
                            real_module = data['module']
                            out = proxy.output
                            is_tuple = isinstance(real_module, torch.nn.MultiheadAttention) or real_module.__class__.__name__ in TUPLE_MODULES
                            corrupted_acts[name] = out[0].save() if is_tuple else out.save()
                    
                    with self.engine.tracer.trace(clean_batch):
                        for name, data in self.engine.target_modules.items():
                            if name not in component_masks:
                                continue
                                
                            proxy = data['proxy']
                            real_module = data['module']
                            out = proxy.output
                            is_tuple = isinstance(real_module, torch.nn.MultiheadAttention) or real_module.__class__.__name__ in TUPLE_MODULES
                            
                            clean_act = out[0] if is_tuple else out
                            corr_act = corrupted_acts[name]
                            mask_tensor = component_masks[name].to(self.device)
                            
                            clean_shape = clean_act.shape
                            mask_shape = mask_tensor.shape
                            
                            if len(clean_shape) == 4 and len(mask_shape) == 2:
                                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(2)
                            elif len(clean_shape) == 4 and len(mask_shape) == 1:
                                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                            elif len(clean_shape) == 3 and len(mask_shape) == 2:
                                mask_tensor = mask_tensor.unsqueeze(1)
                            else:
                                while len(mask_tensor.shape) < len(clean_shape):
                                    mask_tensor = mask_tensor.unsqueeze(0)
                                    
                            if self.model.__class__.__name__ == 'GraphormerModel' or getattr(self.model, 'name', '') == 'graphformer' or 'Graphormer' in real_module.__class__.__name__:
                                mask_tensor = mask_tensor.expand(clean_shape).clone()
                                if len(clean_shape) >= 2:
                                    if clean_shape[1] == batch_size:
                                        mask_tensor[0, ...] = 1.0
                                    elif clean_shape[0] == batch_size:
                                        mask_tensor[:, 0, ...] = 1.0
                            
                            patched_act = (clean_act * mask_tensor) + (corr_act * (1.0 - mask_tensor))
                            
                            if is_tuple:
                                if real_module.__class__.__name__ == 'GraphormerAttentionMessage':
                                    proxy.output = (patched_act, out[1], out[2])
                                else:
                                    proxy.output = (patched_act, out[1])
                            else:
                                proxy.output = patched_act
                                
                        logits = self.engine.tracer.output.logits if hasattr(self.engine.tracer.output, 'logits') else self.engine.tracer.output
                        preds = logits.argmax(dim=-1).save()
                        
                    all_preds.append(preds.cpu())
                    all_targets.append(labels.cpu())
                
                graph_offset += batch_size
                    
        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        
        return f1_score(targets, preds, average='macro', zero_division=0)

    def optimize_graph_circuit(self, edge_attributions: List[Dict[str, torch.Tensor]], save_dir: str):
        """Phase 1: Binary search to find optimal edge pruning threshold."""
        baseline_f1 = self.evaluate_patched_model(component_masks=None, edge_masks=None, dataloader=self.val_dataloader)
        target_f1 = baseline_f1 - self.tolerance
        print(f"Baseline F1: {baseline_f1:.4f} | Target F1: {target_f1:.4f}")

        low, high = 0.0, 100.0
        best_percentile = 0.0
        best_edge_masks = None
        epsilon = 0.5 

        while (high - low) > epsilon:
            mid = (low + high) / 2.0
            edge_masks, _ = self.create_edge_masks_from_percentile(edge_attributions, mid)
            current_f1 = self.evaluate_patched_model(component_masks=None, edge_masks=edge_masks, dataloader=self.val_dataloader)
            
            print(f"Testing Edge Percentile: {mid:5.2f}% | Val F1: {current_f1:.4f}", end="")
            if current_f1 >= target_f1:
                print(" -> PASSED")
                best_percentile = mid
                best_edge_masks = edge_masks
                low = mid
            else:
                print(" -> FAILED")
                high = mid

        print(f"Optimal Graph Circuit Edge Percentile: {best_percentile:.2f}%")
        
        if best_edge_masks is not None:
            mask_path = os.path.join(save_dir, 'optimal_edge_masks.pt')
            torch.save(best_edge_masks, mask_path)
            print(f"Saved optimal edge masks to {mask_path}")
            
        return best_edge_masks, best_percentile

    def optimize(self, attributions: Dict[str, torch.Tensor], save_dir: str, node_attributions: Optional[List[Dict[str, Any]]] = None):
        """Phase 2: Binary search to find optimal model component pruning threshold."""
        baseline_f1 = self.evaluate_patched_model(component_masks=None, edge_masks=None, dataloader=self.val_dataloader)
        target_f1 = baseline_f1 - self.tolerance
        print(f"\nBaseline F1: {baseline_f1:.4f} | Target F1: {target_f1:.4f}")

        low, high = 0.0, 100.0
        best_percentile = 0.0
        best_component_masks = None
        epsilon = 0.5 

        print("--- Starting Binary Search for Model Circuit (Components) ---")
        while (high - low) > epsilon:
            mid = (low + high) / 2.0
            
            # Use safe mask structural reconstruction with component tying properties
            flat_masks, reconstructed_masks, threshold_val = self.create_masks_from_percentile(attributions, mid)
            
            current_f1 = self.evaluate_patched_model(component_masks=reconstructed_masks, edge_masks=None, dataloader=self.val_dataloader)
            
            print(f"Testing Component Percentile: {mid:5.2f}% | Val F1: {current_f1:.4f}", end="")
            if current_f1 >= target_f1:
                print(" -> PASSED")
                best_percentile = mid
                best_component_masks = reconstructed_masks
                low = mid
            else:
                print(" -> FAILED")
                high = mid

        print(f"\nOptimal Model Circuit Component Percentile: {best_percentile:.2f}%")
        
        if best_component_masks is not None:
            mask_path = os.path.join(save_dir, 'optimal_masks.pt')
            torch.save(best_component_masks, mask_path)
            print(f"Saved optimal component masks to {mask_path}")
            
        return best_percentile, best_component_masks

    def run_node_selection_baselines(self, save_dir: str, dataloader = None) -> dict:
        """
        Triggers the GroundTruthEvaluator to compute true causal F1 score drops 
        using Random and Pure Attention node-induced subgraphs.
        """
        if self.gt_evaluator is None:
            raise ValueError("GroundTruthEvaluator is not initialized. Ensure 'config' was passed to ThresholdOptimizer.")
            
        if dataloader is None:
            dataloader = self.test_dataloader

        # Pass the optimizer directly so evaluator can utilize the tracer for pure attention
        return self.gt_evaluator.evaluate_node_selection_baselines(
            save_dir=save_dir,
            dataloader=dataloader,
            model=self.model,
            device=self.device#,
            #optimizer=self # Add this keyword argument to enable on-the-fly execution
        )