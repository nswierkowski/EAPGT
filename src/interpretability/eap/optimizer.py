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
        
        self.engine = eap_engine
        self.dataloader = dataloader
        self.metric_fn = metric_fn
        self.global_scores = global_scores
        
        self.engine.remove_hooks()
        self.baseline_metric = self.metric_fn(self.engine.model, self.dataloader, masks=None)

    def _generate_masks(self, percentile: float) -> Tuple[Dict[str, torch.Tensor], float, Dict[str, float]]:
        """Creates binary masks keeping the top `percentile` of scores independently for Classic and MINAR."""
        masks = {}
        
        # 1. Partition scores into distinct architectural groups
        classic_scores = []
        minar_scores = []
        other_scores = [] # Fallback for components that don't match the standard prefixes
        
        for name, score_tensor in self.global_scores.items():
            if 'classic_' in name:
                classic_scores.append(score_tensor.abs().flatten())
            elif 'minar_' in name:
                minar_scores.append(score_tensor.abs().flatten())
            else:
                other_scores.append(score_tensor.abs().flatten())
                
        # 2. Helper function to safely calculate the cutoff threshold for a group
        def get_group_cutoff(score_list: List[torch.Tensor], p: float) -> float:
            if not score_list:
                return 0.0
            concat_scores = torch.cat(score_list)
            if concat_scores.numel() == 0:
                return 0.0
            return torch.quantile(concat_scores, 1.0 - p).item()

        # 3. Calculate independent cutoffs
        classic_cutoff = get_group_cutoff(classic_scores, percentile)
        minar_cutoff = get_group_cutoff(minar_scores, percentile)
        other_cutoff = get_group_cutoff(other_scores, percentile)
        
        cutoffs = {
            'classic': classic_cutoff,
            'minar': minar_cutoff,
            'other': other_cutoff
        }
        
        total_params = 0
        pruned_params = 0
        
        # 4. Generate masks using the group-specific threshold
        for name, score_tensor in self.global_scores.items():
            if 'classic_' in name:
                cutoff_val = classic_cutoff
            elif 'minar_' in name:
                cutoff_val = minar_cutoff
            else:
                cutoff_val = other_cutoff

            mask = (score_tensor.abs() >= cutoff_val).float()
            masks[name] = mask
            
            total_params += mask.numel()
            pruned_params += (mask == 0).sum().item()
            
        # 5. Calculate global sparsity
        sparsity = pruned_params / total_params if total_params > 0 else 0.0
        return masks, sparsity, cutoffs

    def evaluate_percentile(self, percentile: float) -> Tuple[float, float, Dict[str, float]]:
        """Evaluates model performance at a specific percentile threshold."""
        masks, sparsity, cutoffs = self._generate_masks(percentile)

        patched_metric = self.metric_fn(self.engine.model, self.dataloader, masks=masks)
        
        return patched_metric, sparsity, cutoffs

    def optimize_binary_search(self, max_drop: float, tolerance: float = 0.01) -> Dict:
        """
        Finds the optimal percentile (lowest percentile / highest sparsity) 
        that keeps the metric drop below `max_drop` using Binary Search.
        """
        print(f"Starting Binary Search. Baseline Metric: {self.baseline_metric:.4f}")
        low = 0.0 
        high = 1.0 
        
        best_percentile = 1.0
        best_metric = self.baseline_metric
        best_sparsity = 0.0
        
        target_metric = self.baseline_metric - max_drop

        while (high - low) > tolerance:
            mid = (low + high) / 2.0
            metric, sparsity, cutoffs = self.evaluate_percentile(mid)
            
            # Formatted string to print metrics and group-specific thresholds
            print(f"Testing {mid*100:.1f}% params | F1: {metric:.4f} | Sparsity: {sparsity*100:.1f}% | "
                  f"Cutoffs -> Classic: {cutoffs['classic']:.6f}, MINAR: {cutoffs['minar']:.6f}")
            
            if metric >= target_metric:
                best_percentile = mid
                best_metric = metric
                best_sparsity = sparsity
                high = mid
            else:
                low = mid
                
        print(f"Optimal found: Keep {best_percentile*100:.1f}% params | Sparsity: {best_sparsity*100:.1f}% | F1: {best_metric:.4f}")
        optimal_masks, _, optimal_cutoffs = self._generate_masks(best_percentile)
        
        return {
            'percentile': best_percentile,
            'sparsity': best_sparsity,
            'metric': best_metric,
            'masks': optimal_masks,
            'cutoffs': optimal_cutoffs
        }

    def sweep_curve(self, percentiles: List[float]) -> List[Dict]:
        """
        Evaluates a predefined list of percentiles to generate data for a Pareto curve.
        """
        results = []
        print(f"Starting Sweep. Baseline Metric: {self.baseline_metric:.4f}")
        
        for p in tqdm(sorted(percentiles, reverse=True)):
            metric, sparsity, cutoffs = self.evaluate_percentile(p)
            results.append({
                'percentile': p,
                'sparsity': sparsity,
                'metric': metric,
                'drop': self.baseline_metric - metric,
                'cutoffs': cutoffs
            })
            
        return results