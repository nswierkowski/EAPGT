import torch
import numpy as np
from typing import Dict, List, Tuple, Callable
from tqdm import tqdm

class ThresholdOptimizer:
    def __init__(self, 
                 eap_engine, 
                 dataloader, 
                 metric_fn: Callable, 
                 global_scores: Dict[str, torch.Tensor] = None):
        """
        Args:
            eap_engine: Initialized instance of ClassicEAP, MinarEAP, or HybridEAP.
            dataloader: DataLoader yielding (clean_data, corrupted_data).
            metric_fn: Function that takes (model, dataloader) and returns a scalar metric (e.g., F1).
            global_scores: Precomputed global attribution scores. If None, must be computed later.
        """
        self.engine = eap_engine
        self.dataloader = dataloader
        self.metric_fn = metric_fn
        self.global_scores = global_scores
        
        self.engine.remove_hooks()
        self.baseline_metric = self.metric_fn(self.engine.model, self.dataloader)

    def _generate_masks(self, percentile: float) -> Tuple[Dict[str, torch.Tensor], float]:
        """Creates binary masks keeping only the top `percentile` of scores."""
        masks = {}
        all_scores = []
        
        for score_tensor in self.global_scores.values():
            all_scores.append(score_tensor.abs().flatten())
            
        concat_scores = torch.cat(all_scores)
        
        cutoff_val = torch.quantile(concat_scores, 1.0 - percentile)
        
        total_params = 0
        pruned_params = 0
        
        for name, score_tensor in self.global_scores.items():
            mask = (score_tensor.abs() >= cutoff_val).float()
            masks[name] = mask
            
            total_params += mask.numel()
            pruned_params += (mask == 0).sum().item()
            
        sparsity = pruned_params / total_params if total_params > 0 else 0.0
        return masks, sparsity

    def evaluate_percentile(self, percentile: float) -> Tuple[float, float]:
        """Patches the model at a specific percentile and returns (metric, sparsity)."""
        masks, sparsity = self._generate_masks(percentile)

        self.engine.register_patching_hooks(masks)
        patched_metric = self.metric_fn(self.engine.model, self.dataloader)
        self.engine.remove_hooks()
        
        return patched_metric, sparsity

    def optimize_binary_search(self, max_drop: float, tolerance: float = 0.01) -> Dict:
        """
        Finds the optimal percentile (lowest percentile / highest sparsity) 
        that keeps the metric drop below `max_drop` using Binary Search.
        """
        print(f"Starting Binary Search. Baseline Metric: {self.baseline_metric:.4f}")
        low = 0.0001 
        high = 1.0 
        
        best_percentile = 1.0
        best_metric = self.baseline_metric
        best_sparsity = 0.0
        
        target_metric = self.baseline_metric - max_drop

        while (high - low) > tolerance:
            mid = (low + high) / 2.0
            metric, sparsity = self.evaluate_percentile(mid)
            
            print(f"Testing {mid*100:.1f}% params | F1: {metric:.4f} | Sparsity: {sparsity*100:.1f}%")
            
            if metric >= target_metric:
                best_percentile = mid
                best_metric = metric
                best_sparsity = sparsity
                high = mid
            else:
                low = mid
                
        print(f"Optimal found: Keep {best_percentile*100:.1f}% params | Sparsity: {best_sparsity*100:.1f}% | F1: {best_metric:.4f}")
        optimal_masks, _ = self._generate_masks(best_percentile)
        
        return {
            'percentile': best_percentile,
            'sparsity': best_sparsity,
            'metric': best_metric,
            'masks': optimal_masks
        }

    def sweep_curve(self, percentiles: List[float]) -> List[Dict]:
        """
        Evaluates a predefined list of percentiles to generate data for a Pareto curve.
        """
        results = []
        print(f"Starting Sweep. Baseline Metric: {self.baseline_metric:.4f}")
        
        for p in tqdm(sorted(percentiles, reverse=True)):
            metric, sparsity = self.evaluate_percentile(p)
            results.append({
                'percentile': p,
                'sparsity': sparsity,
                'metric': metric,
                'drop': self.baseline_metric - metric
            })
            
        return results