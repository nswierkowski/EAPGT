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

# Centralize tuple checks like in base.py
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
        # Sample to avoid memory explosion if needed, but let's try full first
        for graph_attr in edge_attributions:
            for score in graph_attr.values():
                all_scores.append(score.flatten())
        
        if not all_scores:
            return [], 0.0
            
        all_scores = torch.cat(all_scores)
        
        # --- PREVENT MEMORY OVERFLOW / QUANTILE LIMITS ---
        # If the tensor is too large (> 5M elements), subsample for threshold estimation
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
                # We want to keep the Head dimension if it exists
                graph_mask[name] = (score >= threshold_val).float()
            edge_masks.append(graph_mask)
            
        return edge_masks, threshold_val

    def evaluate_baseline_pure(self, dataloader=None) -> float:
        """
        Strict Trainer-style evaluation. No NNsight, no complex dictionary unpacking.
        Just standard PyTorch/PyG forward passes to isolate data and model loading issues.
        """
        from sklearn.metrics import precision_recall_fscore_support

        if dataloader is None:
            dataloader = self.val_dataloader

        def _process_batch(batch):
            """Extracts inputs and labels dynamically based on model type."""
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
        """Runs F1 evaluation using NNsight for causal patching instead of manual hooks."""
        if dataloader is None:
            dataloader = self.val_dataloader

        self.model.eval()
        all_preds = []
        all_targets = []
        graph_offset = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating F1", leave=False):
                clean_data = batch['clean']
                if hasattr(clean_data, 'to'):
                    clean_batch = clean_data.to(self.device)
                elif isinstance(clean_data, dict):
                    clean_batch = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in clean_data.items()}
                else:
                    clean_batch = clean_data
                
                labels = clean_batch.y if hasattr(clean_batch, 'y') else clean_batch['labels']
                if labels.dim() > 1 and labels.size(-1) == 1:
                    labels = labels.squeeze(-1)

                if component_masks is None and edge_masks is None:
                    with self.engine.tracer.trace(clean_batch):
                        logits = self.engine.tracer.output.logits if hasattr(self.engine.tracer.output, 'logits') else self.engine.tracer.output
                        preds = logits.argmax(dim=-1).save()
                    
                    all_preds.append(preds.cpu())
                    all_targets.append(labels.cpu())
                    
                else:
                    corr_data = batch['corrupted']
                    if hasattr(corr_data, 'to'):
                        corrupted_batch = corr_data.to(self.device)
                    elif isinstance(corr_data, dict):
                        corrupted_batch = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in corr_data.items()}
                    else:
                        corrupted_batch = corr_data
                    
                    batch_size = labels.size(0)
                    batch_edge_masks = edge_masks[graph_offset : graph_offset + batch_size] if edge_masks is not None else None
                    
                    corrupted_acts = {}
                    
                    # 1. Forward Trace: Extract corrupted batch activations
                    with self.engine.tracer.trace(corrupted_batch):
                        for name, data in self.engine.target_modules.items():
                            proxy = data['proxy']
                            real_module = data['module']
                            
                            out = proxy.output
                            is_tuple_output = isinstance(real_module, torch.nn.MultiheadAttention) or real_module.__class__.__name__ in TUPLE_MODULES
                            
                            if is_tuple_output:
                                out = out[0]
                                    
                            corrupted_acts[name] = out.save()
                    
                    # 2. Forward Trace: Clean batch causal patching
                    with self.engine.tracer.trace(clean_batch):
                        for name, data in self.engine.target_modules.items():
                            proxy = data['proxy']
                            real_module = data['module']
                            out = proxy.output
                            
                            is_tuple_output = isinstance(real_module, torch.nn.MultiheadAttention) or real_module.__class__.__name__ in TUPLE_MODULES
                            
                            clean_act = out[0] if is_tuple_output else out
                            corr_act = corrupted_acts[name] 
                            
                            # Determine which mask to use
                            mask_tensor = None
                            
                            # 1. Check for edge-level masks first (Phase 1 / Graph Circuit)
                            if batch_edge_masks is not None:
                                mask_list = [m[name] for m in batch_edge_masks if name in m]
                                if mask_list:
                                    mask_tensor = torch.stack(mask_list).to(self.device)
                            
                            # 2. Check for component-level masks (Phase 2 / Model Circuit)
                            if mask_tensor is None and component_masks is not None and name in component_masks:
                                mask_tensor = component_masks[name].to(self.device)
                            
                            if mask_tensor is None:
                                continue
                                
                            clean_shape = clean_act.shape
                            mask_shape = mask_tensor.shape
                            
                            # Safely broadcast mask
                            if len(clean_shape) == 4 and len(mask_shape) == 2:
                                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(2)
                            elif len(clean_shape) == 4 and len(mask_shape) == 1:
                                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                            elif len(clean_shape) == 3 and len(mask_shape) == 2:
                                mask_tensor = mask_tensor.unsqueeze(1)
                            else:
                                while len(mask_tensor.shape) < len(clean_shape):
                                    mask_tensor = mask_tensor.unsqueeze(0)
                            
                            # VNode patch protection for Graphormer (modify mask_tensor safely instead of Proxy)
                            if self.model.__class__.__name__ == 'GraphormerModel' or getattr(self.model, 'name', '') == 'graphformer' or 'Graphormer' in real_module.__class__.__name__:
                                mask_tensor = mask_tensor.expand(clean_shape).clone()
                                if len(clean_shape) >= 2:
                                    # HF Graphormer uses seq_first (seq_len, batch_size, hidden_size) in its encoder
                                    if clean_shape[1] == batch_size:
                                        mask_tensor[0, ...] = 1.0
                                    elif clean_shape[0] == batch_size:
                                        mask_tensor[:, 0, ...] = 1.0
                            
                            # Apply the patch
                            patched_act = (clean_act * mask_tensor) + (corr_act * (1.0 - mask_tensor))
                            

                            if is_tuple_output:
                                if real_module.__class__.__name__ == 'GraphormerAttentionMessage':
                                    # Graphormer's message module returns 3 items: (attn_probs, v, is_seq_first)
                                    proxy.output = (patched_act, out[1], out[2])
                                else:
                                    # Standard MultiheadAttention returns 2 items: (output, weights)
                                    proxy.output = (patched_act, out[1])
                            else:
                                proxy.output = patched_act
                                
                        logits = self.engine.tracer.output.logits if hasattr(self.engine.tracer.output, 'logits') else self.engine.tracer.output
                        preds = logits.argmax(dim=-1).save()
                        
                    all_preds.append(preds.cpu())
                    all_targets.append(labels.cpu())
                
                graph_offset += labels.size(0)
                    
        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        
        from sklearn.metrics import f1_score
        return f1_score(targets, preds, average='macro', zero_division=0)

    def extract_and_plot_optimal_circuit(self, attributions: Dict[str, torch.Tensor], threshold_percentile: float, save_dir: str) -> nx.DiGraph:
        """Builds the final network graph, handling parallel multi-head logic safely."""
        print(f"Extracting Optimal Circuit (Threshold: {threshold_percentile:.2f}th percentile)...")
        
        all_scores = torch.cat([score.flatten() for score in attributions.values()])
        threshold_val = torch.quantile(all_scores.float(), threshold_percentile / 100.0).item()
        
        G = nx.DiGraph()
        ordered_components = list(attributions.keys())
        
        for comp in ordered_components:
            score_max = attributions[comp].max().item()
            if score_max >= threshold_val:
                G.add_node(comp, eap_importance=score_max)

        for i in range(len(ordered_components) - 1):
            src = ordered_components[i]
            dst = ordered_components[i+1]
            
            if ".head_" in src and ".head_" in dst:
                if src.split(".head_")[0] == dst.split(".head_")[0]:
                    continue
                    
            if ".head_" in src and dst.endswith(".A"):
                if G.has_node(src) and G.has_node(dst):
                    G.add_edge(src, dst, weight=attributions[dst].max().item())
                continue
                
            if G.has_node(src) and G.has_node(dst):
                G.add_edge(src, dst, weight=attributions[dst].max().item())

        os.makedirs(save_dir, exist_ok=True)
        graph_path = os.path.join(save_dir, "optimal_macro_circuit.pkl")
        with open(graph_path, "wb") as f:
            pickle.dump(G, f)
        
        print(f"Optimal circuit generated with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G

    def optimize(self, attributions: Dict[str, torch.Tensor], save_dir: str, node_attributions: Optional[List[Dict]] = None):
        """Executes the binary search for the optimal pruning threshold on the validation set,
        then validates the final circuit against the full model on the test set.
        
        If node_attributions is provided, also evaluates faithfulness to ground truth.
        """
        
        print("Calculating Pure Trainer Baseline (Validation)...")
        pure_f1 = self.evaluate_baseline_pure(dataloader=self.val_dataloader)
        
        print("Calculating Baseline F1 - No Pruning (Validation)...")
        baseline_f1 = self.evaluate_patched_model(component_masks=None, dataloader=self.val_dataloader)
        
        print(f"\n--- VALIDATION BASELINE COMPARISON ---")
        print(f"Pure Trainer F1: {pure_f1 * 100:.2f}%")
        print(f"Optimizer F1:    {baseline_f1 * 100:.2f}%")
                
        target_f1 = baseline_f1 - self.tolerance
        print(f"Baseline F1: {baseline_f1:.4f} | Target F1 (>{self.tolerance*100}% drop): {target_f1:.4f}")

        low, high = 0.0, 100.0
        best_percentile, best_f1 = 0.0, baseline_f1
        best_flat_masks = None
        best_recon_masks = None
        epsilon = 0.5 

        print("\n--- Starting Binary Search on Validation Set ---")
        while (high - low) > epsilon:
            mid_percentile = (low + high) / 2.0
            
            flat_masks, recon_masks, _ = self.create_masks_from_percentile(attributions, mid_percentile)
            current_f1 = self.evaluate_patched_model(component_masks=recon_masks, dataloader=self.val_dataloader)
            
            print(f"Testing Percentile: {mid_percentile:5.2f}% | Val F1: {current_f1:.4f}", end="")
            
            if current_f1 >= target_f1:
                print(" -> PASSED! Pruning more.")
                best_percentile = mid_percentile
                best_flat_masks = flat_masks
                best_recon_masks = recon_masks
                best_f1 = current_f1
                low = mid_percentile
            else:
                print(" -> FAILED! Pruning less.")
                high = mid_percentile

        print(f"\n--- Validation Search Complete ---")
        print(f"Optimal Pruning Percentile: {best_percentile:.2f}%")
        print(f"Final Patched Val F1 Score: {best_f1:.4f} (Drop: {baseline_f1 - best_f1:.4f})")

        print("\n--- Final Test Set Evaluation ---")
        test_full_f1 = self.evaluate_patched_model(component_masks=None, dataloader=self.test_dataloader)
        print(f"Full Model Test F1:  {test_full_f1:.4f}")

        test_circuit_f1 = None
        if best_recon_masks is not None:
            test_circuit_f1 = self.evaluate_patched_model(component_masks=best_recon_masks, dataloader=self.test_dataloader)
            print(f"Circuit Test F1:     {test_circuit_f1:.4f}")
            print(f"Test Set Drop:       {test_full_f1 - test_circuit_f1:.4f}")

            mask_path = os.path.join(save_dir, 'optimal_masks.pt')
            torch.save(best_flat_masks, mask_path)
            
            print("\n--- Kept Components in Optimal Circuit ---")
            kept_components = []
            for name, mask in best_flat_masks.items():
                if mask.max() > 0:
                    score = attributions[name].max().item()
                    kept_components.append((name, score))
            
            # Sort by importance
            kept_components.sort(key=lambda x: x[1], reverse=True)
            for name, score in kept_components:
                print(f" - {name:40} | Score: {score:.6f}")
                
            self.extract_and_plot_optimal_circuit(attributions, best_percentile, save_dir)
        else:
            print("Warning: Could not find any threshold that meets the tolerance. Skipping test evaluation.")

        results = {
            "optimal_threshold_percentile": float(best_percentile),
            "val_baseline_f1": float(baseline_f1),
            "val_circuit_f1": float(best_f1),
            "test_baseline_f1": float(test_full_f1),
            "test_circuit_f1": float(test_circuit_f1) if test_circuit_f1 is not None else None,
            "tolerance": self.tolerance,
        }

        os.makedirs(save_dir, exist_ok=True)
        results_path = os.path.join(save_dir, "optimization_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"\nSaved optimization results to: {results_path}")

        if node_attributions is not None and self.gt_evaluator is not None:
            print("\n--- Ground Truth Faithfulness Evaluation ---")
            
            def run_faithfulness_eval(node_attrs, threshold_percentile=None):
                all_metrics = []
                
                masks = None
                if threshold_percentile is not None:
                    masks, _, _ = self.create_masks_from_percentile(attributions, threshold_percentile)

                for entry in tqdm(node_attrs, desc="Evaluating Faithfulness", leave=False):
                    from torch_geometric.data import Data
                    dummy_data = Data(x=entry.get('x'), edge_index=entry.get('edge_index'), y=entry.get('y'))
                    gt_mask = self.gt_evaluator.get_gt_mask(dummy_data)
                    raw_node_scores = entry['node_scores']
                    
                    if masks is not None:
                        pruned_node_scores = {}
                        for k, v in raw_node_scores.items():
                            mask_key = k
                            if mask_key not in masks and '.head_' in mask_key:
                                mask_key = mask_key.split('.head_')[0]
                                
                            if mask_key in masks and masks[mask_key].max() > 0:
                                pruned_node_scores[k] = v
                                
                        if not pruned_node_scores:
                            scalar_scores = torch.zeros(gt_mask.shape)
                        else:
                            scalar_scores = self.gt_evaluator.aggregate_node_scores(pruned_node_scores, num_nodes=gt_mask.shape[0])
                    else:
                        scalar_scores = self.gt_evaluator.aggregate_node_scores(raw_node_scores, num_nodes=gt_mask.shape[0])
                    
                    metrics = self.gt_evaluator.calculate_faithfulness_metrics(scalar_scores, gt_mask)
                    if metrics:
                        all_metrics.append(metrics)
                
                if not all_metrics:
                    print("Warning: No valid GT samples found for faithfulness evaluation.")
                    return {k: 0.0 for k in ["auprc", "precision_at_k", "attribution_gap", "mean_gt_score", "mean_non_gt_score"]}

                df = pd.DataFrame(all_metrics)
                print(f" (Evaluated on {len(all_metrics)} valid samples)", end="")
                return df.mean().to_dict()

            print("Evaluating Full EAP Distribution...")
            full_faithfulness = run_faithfulness_eval(node_attributions)
            
            print(f"Evaluating Pruned Model (Threshold: {best_percentile:.2f}%)...")
            pruned_faithfulness = run_faithfulness_eval(node_attributions, threshold_percentile=best_percentile)
            
            comparison = {
                "Metric": ["AUPRC", "Precision@K", "Attribution Gap", "Mean GT Score", "Mean Non-GT Score"],
                "Full EAP": [
                    full_faithfulness['auprc'], 
                    full_faithfulness['precision_at_k'], 
                    full_faithfulness['attribution_gap'],
                    full_faithfulness['mean_gt_score'],
                    full_faithfulness['mean_non_gt_score']
                ],
                "Pruned Circuit": [
                    pruned_faithfulness['auprc'], 
                    pruned_faithfulness['precision_at_k'], 
                    pruned_faithfulness['attribution_gap'],
                    pruned_faithfulness['mean_gt_score'],
                    pruned_faithfulness['mean_non_gt_score']
                ]
            }
            comp_df = pd.DataFrame(comparison)
            print("\n--- Circuit Faithfulness Comparison ---")
            print(comp_df.to_string(index=False))
            
            comp_df.to_csv(os.path.join(save_dir, "faithfulness_comparison.csv"), index=False)
            
            def get_all_scores_and_masks(node_attrs, threshold_percentile=None):
                all_s = []
                all_m = []
                
                masks = None
                if threshold_percentile is not None:
                    masks, _, _ = self.create_masks_from_percentile(attributions, threshold_percentile)

                for entry in node_attrs:
                    from torch_geometric.data import Data
                    dummy_data = Data(x=entry.get('x'), edge_index=entry.get('edge_index'), y=entry.get('y'))
                    gt_mask = self.gt_evaluator.get_gt_mask(dummy_data)
                    raw_node_scores = entry['node_scores']
                    if masks is not None:
                        pruned_node_scores = {}
                        for k, v in raw_node_scores.items():
                            mask_key = k
                            if mask_key not in masks and '.head_' in mask_key:
                                mask_key = mask_key.split('.head_')[0]
                            if mask_key in masks and masks[mask_key].max() > 0:
                                pruned_node_scores[k] = v
                        scalar_scores = self.gt_evaluator.aggregate_node_scores(pruned_node_scores, num_nodes=gt_mask.shape[0]) if pruned_node_scores else torch.zeros_like(gt_mask)
                    else:
                        scalar_scores = self.gt_evaluator.aggregate_node_scores(raw_node_scores, num_nodes=gt_mask.shape[0])
                    
                    if gt_mask.sum() > 0:
                        all_s.append(scalar_scores.detach().cpu())
                        all_m.append(gt_mask.detach().cpu())
                
                if not all_s:
                    return torch.zeros(1), torch.zeros(1)
                    
                return torch.cat(all_s), torch.cat(all_m)

            print("Generating Attribution Gap plots...")
            full_s, full_m = get_all_scores_and_masks(node_attributions)
            self.gt_evaluator.plot_attribution_gap(full_s, full_m, os.path.join(save_dir, "attribution_gap_full.png"), "(Full EAP)")
            pruned_s, pruned_m = get_all_scores_and_masks(node_attributions, threshold_percentile=best_percentile)
            self.gt_evaluator.plot_attribution_gap(pruned_s, pruned_m, os.path.join(save_dir, "attribution_gap_pruned.png"), f"(Pruned @ {best_percentile:.2f}%)")

    def optimize_graph_circuit(self, edge_attributions: List[Dict[str, torch.Tensor]], save_dir: str) -> List[Dict[str, torch.Tensor]]:
        """
        Executes binary search for the optimal edge threshold (Phase 1).
        """
        print("\n--- Phase 1: Graph-Level Circuit Discovery (Topology) ---")
        
        baseline_f1 = self.evaluate_patched_model(component_masks=None, edge_masks=None, dataloader=self.val_dataloader)
        target_f1 = baseline_f1 - self.tolerance
        print(f"Baseline F1: {baseline_f1:.4f} | Target F1: {target_f1:.4f}")

        low, high = 0.0, 100.0
        best_percentile = 0.0
        best_edge_masks = None
        epsilon = 1.0 

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
        
        if best_edge_masks:
            mask_path = os.path.join(save_dir, 'optimal_edge_masks.pt')
            # Extract just a representative mask or enough for the circuit visualization
            # Saving the whole list might be too large for .pt if not careful
            torch.save(best_edge_masks, mask_path)
            print(f"Saved optimal edge masks to {mask_path}")
            
        return best_edge_masks, best_percentile