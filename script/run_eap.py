import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader 

from src.models.graphgps.model import GraphGPSModel 
from src.models.graphformer.model import GraphormerModel
from src.interpretability.eap.strategies import ClassicEAP, MinarEAP, HybridEAP
from src.interpretability.eap.optimizer import ThresholdOptimizer

from src.interpretability.counterfactuals.factory import get_counterfactual_dataset

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
        clean_batch, corrupted_batch = batch['clean'].to(device), batch['corrupted'].to(device)
        
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

def create_patched_metric_fn(engine, device):
    """
    Creates a metric function that is EAP-aware.
    To patch, the engine needs A_corrupted in its cache before running the clean pass.
    """
    def metric_fn(model, dataloader):
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                clean_batch = batch['clean'].to(device)
                corrupted_batch = batch['corrupted'].to(device)
                
                engine.register_corrupted_hooks()
                model(corrupted_batch)
                
                preds = model(clean_batch).argmax(dim=-1)
                
                all_preds.append(preds.cpu())
                all_targets.append(clean_batch.y.cpu())
                
        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        return f1_score(targets, preds, average='macro')
        
    return metric_fn

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
    
    dataloader = DataLoader(
        cf_data_list, 
        batch_size=config['dataset']['batch_size'], 
        shuffle=False 
    )

    model = get_model(config).to(device)
    loss_fn = nn.CrossEntropyLoss()

    engine = get_eap_engine(config['eap']['strategy'], model, config)

    global_scores = compute_global_scores(engine, dataloader, loss_fn, device)

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