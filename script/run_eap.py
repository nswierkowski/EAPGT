import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader 
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.graphgps.model import GraphGPSModel 
from src.models.graphformer.model import GraphormerModel
from src.interpretability.eap.strategies import ClassicEAP, MinarEAP, HybridEAP
from src.interpretability.eap.optimizer import ThresholdOptimizer
from src.interpretability.counterfactuals.factory import get_counterfactual_dataset

from src.data.collator import GraphTransformerCollator

def set_seed(seed: int):
    """Ensures complete reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(config):
    if config['model']['architecture'] == 'graphgps':
        return GraphGPSModel(config)
    elif config['model']['architecture'] == 'graphormer':
        return GraphormerModel(config)
    else:
        raise ValueError(f"Unknown architecture: {config['model']['architecture']}")

def get_eap_engine(strategy_str: str, model: nn.Module, config: dict):
    strategies = {
        'classic': ClassicEAP,
        'minar': MinarEAP,
        'hybrid': HybridEAP
    }
    return strategies[strategy_str](model, config)

def compute_global_scores(engine, dataloader, loss_fn, device) -> dict:
    """Aggregates EAP scores across the entire dataset to find global importance."""
    print("Computing global attribution scores...")
    global_scores = {}
    total_graphs = 0
    
    for batch in tqdm(dataloader, desc="EAP Scoring"):
        clean_batch = batch['clean'].to(device) if hasattr(batch['clean'], 'to') else batch['clean']
        corrupted_batch = batch['corrupted'].to(device) if hasattr(batch['corrupted'], 'to') else batch['corrupted']
        
        batch_scores = engine.evaluate_pair(clean_batch, corrupted_batch, loss_fn)
        
        for name, score_tensor in batch_scores.items():
            feature_score = score_tensor.abs().sum(dim=0)
            
            if name not in global_scores:
                global_scores[name] = torch.zeros_like(feature_score).float()

            global_scores[name] += feature_score
            
        total_graphs += clean_batch.num_graphs if hasattr(clean_batch, 'num_graphs') else 1

    for name in global_scores.keys():
        global_scores[name] /= total_graphs
        
    return global_scores

def get_cf_collate_fn(config):
    base_collator = GraphTransformerCollator(config)
    
    def collate_fn(batch_list):
        clean_list = [item['clean'] for item in batch_list]
        corrupted_list = [item['corrupted'] for item in batch_list]
        
        return {
            'clean': base_collator(clean_list),
            'corrupted': base_collator(corrupted_list)
        }
    return collate_fn

def create_patched_metric_fn(engine, device):
    """
    Creates a metric function that handles per-batch corrupted caching AND patching.
    Takes 'masks' as an optional argument.
    """
    def metric_fn(model, dataloader, masks=None):
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                clean_batch = batch['clean'].to(device) if hasattr(batch['clean'], 'to') else batch['clean']
                corrupted_batch = batch['corrupted'].to(device) if hasattr(batch['corrupted'], 'to') else batch['corrupted']
                
                engine.register_corrupted_hooks()
                model(corrupted_batch)
                
                if masks is not None:
                    engine.register_patching_hooks(masks)
                else:
                    engine.remove_hooks() 
                
                # IMPERATIVE FIX: Align prediction and label extraction with your Trainer
                outputs = model(clean_batch)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                preds = logits.argmax(dim=-1)
                
                labels = clean_batch.y if hasattr(clean_batch, 'y') else clean_batch['labels']
                if labels.dim() > 1 and labels.size(-1) == 1:
                    labels = labels.squeeze(-1)
                
                all_preds.append(preds.cpu())
                all_targets.append(labels.cpu())
                
        engine.remove_hooks()
                
        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        return f1_score(targets, preds, average='macro')
        
    return metric_fn

def analyze_attribution_scores(global_scores, save_dir, strategy_name):
    """Saves raw EAP scores and plots their distribution."""
    print(f"Saving and analyzing {strategy_name.upper()} attribution scores...")

    torch.save(global_scores, os.path.join(save_dir, 'global_attributions.pt'))

    all_scores = []
    for name, score_tensor in global_scores.items():
        all_scores.append(score_tensor.detach().cpu().flatten().abs())
        
    if not all_scores:
        print("Warning: No attribution scores found to plot.")
        return

    flat_scores = torch.cat(all_scores).numpy()

    plt.figure(figsize=(10, 6))
    
    sns.histplot(flat_scores, bins=100, log_scale=(False, True), color='#9b59b6', 
                     label=f"{strategy_name.capitalize()} Scores\nMax: {flat_scores.max():.4f}\nMean: {flat_scores.mean():.6f}")
    plt.title(f"EAP Attribution Score Distribution ({strategy_name.capitalize()})", fontsize=14)
    plt.xlabel("Absolute Attribution Score |Act_diff * Grad|")
    plt.ylabel("Count (Log Scale)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(save_dir, 'attribution_distribution.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Saved attribution data to {save_dir}/global_attributions.[pt|png]")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to EAP YAML config")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['experiment']['random_state'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config['experiment']['save_dir'], exist_ok=True)
    print(f"Loading {config['dataset']['name']} dataset and counterfactuals...")
    
    base_dataset = None 
    cf_data_list = get_counterfactual_dataset(config, base_dataset=base_dataset)
    
    cf_collate_fn = get_cf_collate_fn(config)
    dataloader = DataLoader(
        cf_data_list, 
        batch_size=config['dataset']['batch_size'], 
        shuffle=False,
        collate_fn=cf_collate_fn
    )

    model = get_model(config).to(device)
    checkpoint_path = config['model'].get('checkpoint_path')
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded trained checkpoint from {checkpoint_path}")
    else:
        print("WARNING: No checkpoint found! Running on random weights.")

    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    engine = get_eap_engine(config['eap']['strategy'], model, config)

    global_scores = compute_global_scores(engine, dataloader, loss_fn, device)

    analyze_attribution_scores(global_scores, config['experiment']['save_dir'], config['eap']['strategy'])

    metric_fn = create_patched_metric_fn(engine, device)
    optimizer = ThresholdOptimizer(engine, dataloader, metric_fn, global_scores)

    if config['optimizer']['mode'] == 'binary_search':
        results = optimizer.optimize_binary_search(
            max_drop=config['optimizer']['max_degradation'],
            tolerance=config['optimizer']['tolerance']
        )
        
        torch.save(results['masks'], os.path.join(config['experiment']['save_dir'], 'optimal_masks.pt'))
        print(f"Successfully saved optimal masks to {config['experiment']['save_dir']}")
        
    elif config['optimizer']['mode'] == 'sweep':
        results = optimizer.sweep_curve(config['optimizer']['sweep_percentiles'])
        torch.save(results, os.path.join(config['experiment']['save_dir'], 'pareto_curve_data.pt'))

if __name__ == "__main__":
    main()