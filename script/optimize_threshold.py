import os
import yaml
import torch
import argparse
from torch_geometric.loader import DataLoader

from src.models.wrapper import instrument_model
from src.models.graphgps.model import GraphGPSModel 
from src.models.graphformer.model import GraphormerModel
from src.interpretability.eap.strategies import MacroMinarEAP
from src.interpretability.counterfactuals.factory import get_counterfactual_dataset
from src.data.collator import GraphTransformerCollator
from src.interpretability.eap.optimizer import ThresholdOptimizer

def get_model(config):
    if config['model']['architecture'] == 'graphgps':
        return GraphGPSModel(config)
    elif config['model']['architecture'] == 'graphormer':
        return GraphormerModel(config)
    raise ValueError(f"Unknown architecture: {config['model']['architecture']}")

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to EAP YAML config")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Max allowed drop in F1")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = config['experiment']['save_dir']

    cf_data_list = get_counterfactual_dataset(config, base_dataset=None)

    val_cf_list = [
        cf_pair for cf_pair in cf_data_list
        if hasattr(cf_pair['clean'], 'split_mask') and cf_pair['clean'].split_mask.item() == 1
    ]
    test_cf_list = [
        cf_pair for cf_pair in cf_data_list
        if hasattr(cf_pair['clean'], 'split_mask') and cf_pair['clean'].split_mask.item() == 2
    ]

    if not val_cf_list:
        raise ValueError("No validation samples found (split_mask == 1). Ensure run_eap.py generated CFs for the val split.")
    if not test_cf_list:
        raise ValueError("No test samples found (split_mask == 2). Ensure run_eap.py generated CFs for the test split.")

    val_dataloader = DataLoader(
        val_cf_list,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        collate_fn=get_cf_collate_fn(config)
    )
    test_dataloader = DataLoader(
        test_cf_list,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        collate_fn=get_cf_collate_fn(config)
    )

    model = get_model(config).to(device)
    checkpoint = torch.load(config['model']['checkpoint_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    target_classes = config['model'].get('target_wrapper_classes', [])
    model = instrument_model(model, target_classes)
    
    engine = MacroMinarEAP(model, config) 
    
    attr_path = os.path.join(save_dir, 'global_attributions.pt')
    if not os.path.exists(attr_path):
         raise FileNotFoundError(f"Run run_eap.py first! Could not find {attr_path}")
    attributions = torch.load(attr_path, map_location=device)

    optimizer = ThresholdOptimizer(
        engine, model, val_dataloader, test_dataloader, device, tolerance=args.tolerance
    )
    optimizer.optimize(attributions, save_dir)

if __name__ == "__main__":
    main()