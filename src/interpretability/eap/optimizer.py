import os
import json
import torch
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score
from typing import Dict, Any
from src.models.wrappers.attention_mpnn import AttentionMessage
from src.models.wrappers.base import BaseMPNNWrapper

class ThresholdOptimizer:
    def __init__(self, engine, model, val_dataloader, test_dataloader, device, tolerance: float = 0.05):
        self.engine = engine
        self.model = model
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.tolerance = tolerance

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
            
        reconstructed_masks = {}
        head_groups = {}
        
        for name, mask in flat_masks.items():
            if ".head_" in name:
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

    def evaluate_patched_model(self, masks: Dict[str, torch.Tensor] = None, dataloader=None) -> float:
        """Runs F1 evaluation using NNsight for causal patching instead of manual hooks."""
        if dataloader is None:
            dataloader = self.val_dataloader

        self.model.eval()
        all_preds = []
        all_targets = []
        
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

                if masks is None:
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

                    corrupted_acts = {}
                    
                    with self.engine.tracer.trace(corrupted_batch):
                        for name, data in self.engine.target_modules.items():
                            proxy = data['proxy']
                            real_module = data['module']
                            
                            out = proxy.output
                            if isinstance(real_module, (torch.nn.MultiheadAttention, AttentionMessage)):
                                out = out[0]
                                    
                            corrupted_acts[name] = out.save()
                    
                    with self.engine.tracer.trace(clean_batch):
                        for name, data in self.engine.target_modules.items():
                            if name not in masks:
                                continue
                                
                            proxy = data['proxy']
                            real_module = data['module']
                            out = proxy.output
                            
                            is_tuple_output = isinstance(real_module, (torch.nn.MultiheadAttention, AttentionMessage))
                            clean_act = out[0] if is_tuple_output else out
                            corr_act = corrupted_acts[name] 
                            mask_tensor = masks[name].to(self.device)
                            
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
                            
                            patched_act = (clean_act * mask_tensor) + (corr_act * (1.0 - mask_tensor))
                            
                            if is_tuple_output:
                                proxy.output = (patched_act, out[1]) 
                            else:
                                proxy.output = patched_act
                                
                        logits = self.engine.tracer.output.logits if hasattr(self.engine.tracer.output, 'logits') else self.engine.tracer.output
                        preds = logits.argmax(dim=-1).save()
                        
                    all_preds.append(preds.cpu())
                    all_targets.append(labels.cpu())
                    
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

    def optimize(self, attributions: Dict[str, torch.Tensor], save_dir: str):
        """Executes the binary search for the optimal pruning threshold on the validation set,
        then validates the final circuit against the full model on the test set."""
        
        print("Calculating Pure Trainer Baseline (Validation)...")
        pure_f1 = self.evaluate_baseline_pure(dataloader=self.val_dataloader)
        
        print("Calculating Baseline F1 - No Pruning (Validation)...")
        baseline_f1 = self.evaluate_patched_model(masks=None, dataloader=self.val_dataloader)
        
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
            current_f1 = self.evaluate_patched_model(masks=recon_masks, dataloader=self.val_dataloader)
            
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
        test_full_f1 = self.evaluate_patched_model(masks=None, dataloader=self.test_dataloader)
        print(f"Full Model Test F1:  {test_full_f1:.4f}")

        test_circuit_f1 = None
        if best_recon_masks is not None:
            test_circuit_f1 = self.evaluate_patched_model(masks=best_recon_masks, dataloader=self.test_dataloader)
            print(f"Circuit Test F1:     {test_circuit_f1:.4f}")
            print(f"Test Set Drop:       {test_full_f1 - test_circuit_f1:.4f}")

            mask_path = os.path.join(save_dir, 'optimal_masks.pt')
            torch.save(best_flat_masks, mask_path)
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