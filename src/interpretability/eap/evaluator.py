import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc
import os

class GroundTruthEvaluator:
    def __init__(self, dataset_name: str, config: dict):
        self.dataset_name = dataset_name.lower()
        self.config = config

    def get_gt_mask(self, data) -> torch.Tensor:
        """Extracts ground truth node mask for a single graph sample."""
        if 'ba_shapes' in self.dataset_name:
            return self._get_ba_shapes_mask(data)
        elif 'zinc' in self.dataset_name:
            return self._get_zinc_mask(data)
        else:
            raise ValueError(f"Unsupported dataset for GT evaluation: {self.dataset_name}")

    def _get_ba_shapes_mask(self, data) -> torch.Tensor:
        """
        For BA-Shapes: nodes in the 'house' motif are marked 1.
        Supports both node labels (if data.y is a vector) and graph labels (last 5 nodes).
        """
        num_nodes = data.x.shape[0]
        mask = torch.zeros(num_nodes, dtype=torch.float)
        
        # Scenario A: Node labels are provided (labels 1, 2, 3 are the house)
        if hasattr(data, 'y') and data.y.dim() > 0 and data.y.shape[0] == num_nodes:
            return ((data.y >= 1) & (data.y <= 3)).float()
            
        # Scenario B: Graph labels are provided (label 1 means house exists at indices [-5:])
        if hasattr(data, 'y') and data.y.item() == 1:
            mask[-5:] = 1.0
        
        return mask

    def _get_zinc_mask(self, data) -> torch.Tensor:
        """
        For ZINC: Identify NO2 and NH2 groups based on atomic numbers and connectivity.
        N=7, O=8, H=1.
        """
        num_nodes = data.x.shape[0]
        mask = torch.zeros(num_nodes, dtype=torch.float)
        
        if not hasattr(data, 'edge_index'):
            return mask
            
        atomic_nums = data.x.flatten()
        edge_index = data.edge_index
        
        # Simple adjacency list for connectivity
        adj = [[] for _ in range(num_nodes)]
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            adj[u].append(v)
            
        for i in range(num_nodes):
            if atomic_nums[i] == 7: # Nitrogen
                neighbors = adj[i]
                neighbor_atoms = atomic_nums[neighbors]
                
                # Check for NO2 (N connected to at least 2 O)
                num_o = (neighbor_atoms == 8).sum().item()
                # Check for NH2 (N connected to at least 2 H)
                num_h = (neighbor_atoms == 1).sum().item()
                
                if num_o >= 2 or num_h >= 2:
                    mask[i] = 1.0
                    # Also mark the neighbors as part of the group
                    for n in neighbors:
                        if (num_o >= 2 and atomic_nums[n] == 8) or (num_h >= 2 and atomic_nums[n] == 1):
                            mask[n] = 1.0
                            
        return mask

    def aggregate_node_scores(self, node_attributions: dict, num_nodes: int = None) -> torch.Tensor:
        """
        Aggregates scores from all layers and heads to produce a single scalar importance score per node.
        S_v = sum_{layer} sum_{head} score(v, source).
        """
        if not node_attributions:
            if num_nodes is not None:
                return torch.zeros(num_nodes)
            return torch.zeros(1) # Fallback

        # Get any score tensor to find number of nodes if not provided
        if num_nodes is None:
            first_key = list(node_attributions.keys())[0]
            first_score = node_attributions[first_key]
            num_nodes = first_score.shape[-1]
            device = first_score.device
        else:
            # Find a device from any entry if possible
            device = next(iter(node_attributions.values())).device if node_attributions else torch.device('cpu')
        
        total_score = torch.zeros(num_nodes, device=device)
        
        for name, scores in node_attributions.items():
            # scores is (heads, num_nodes) or (num_nodes,)
            # If scores have VNode (N+1), but total_score is N, we strip the VNode (index 0)
            cur_scores = scores
            if cur_scores.shape[-1] == num_nodes + 1:
                cur_scores = cur_scores[..., 1:]
            elif cur_scores.shape[-1] != num_nodes:
                # Handle edge cases/padding: just take first num_nodes if bigger, or skip if smaller
                if cur_scores.shape[-1] > num_nodes:
                    cur_scores = cur_scores[..., :num_nodes]
                else:
                    continue 

            if cur_scores.dim() > 1:
                total_score += cur_scores.sum(dim=0)
            else:
                total_score += cur_scores
                
        return total_score

    def calculate_faithfulness_metrics(self, scores: torch.Tensor, gt_mask: torch.Tensor) -> dict:
        """
        Computes AUPRC, Precision@K, and Attribution Gap.
        """
        scores_np = scores.detach().cpu().numpy()
        gt_mask_np = gt_mask.detach().cpu().numpy()
        
        # Check if there are any positive samples in gt_mask.
        # If not, this graph is not suitable for faithfulness evaluation (no motif to find).
        if gt_mask_np.sum() == 0:
            return None

        # AUPRC
        precision, recall, _ = precision_recall_curve(gt_mask_np, scores_np)
        auprc = auc(recall, precision)
        
        # Precision@K (K = number of nodes in GT motif)
        k = int(gt_mask_np.sum())
        if k > 0:
            top_k_indices = np.argsort(scores_np)[-k:]
            precision_at_k = gt_mask_np[top_k_indices].mean()
        else:
            precision_at_k = 0.0
            
        # Attribution Gap: Mean(Scores of GT Nodes) / Mean(Scores of Non-GT Nodes)
        gt_scores = scores_np[gt_mask_np == 1]
        non_gt_scores = scores_np[gt_mask_np == 0]
        
        mean_gt = gt_scores.mean() if len(gt_scores) > 0 else 0.0
        mean_non_gt = non_gt_scores.mean() if len(non_gt_scores) > 0 else 1e-9
        attribution_gap = mean_gt / (mean_non_gt + 1e-9)
        
        return {
            "auprc": float(auprc),
            "precision_at_k": float(precision_at_k),
            "attribution_gap": float(attribution_gap),
            "mean_gt_score": float(mean_gt),
            "mean_non_gt_score": float(mean_non_gt)
        }

    def plot_attribution_gap(self, scores: torch.Tensor, gt_mask: torch.Tensor, save_path: str, title_suffix: str = ""):
        """Plots the EAP score distribution for GT vs. Non-GT nodes."""
        scores_np = scores.detach().cpu().numpy()
        gt_mask_np = gt_mask.detach().cpu().numpy()
        
        plt.figure(figsize=(10, 6))
        
        gt_scores = scores_np[gt_mask_np == 1]
        non_gt_scores = scores_np[gt_mask_np == 0]
        
        if len(gt_scores) > 0:
            sns.kdeplot(gt_scores, label='Ground Truth Nodes', fill=True, color='green')
        if len(non_gt_scores) > 0:
            sns.kdeplot(non_gt_scores, label='Non-GT Nodes', fill=True, color='red')
            
        plt.title(f"Node Attribution Distribution {title_suffix}")
        plt.xlabel("Aggregation EAP Score")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
