import os
import yaml
import torch
import argparse
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader

from src.models.wrapper import instrument_model
from src.models.graphgps.model import GraphGPSModel 
from src.models.graphformer.model import GraphormerModel
from src.interpretability.eap.strategies import MacroMinarEAP
from src.interpretability.counterfactuals.factory import get_counterfactual_dataset
from src.data.collator import GraphTransformerCollator
from src.interpretability.eap.optimizer import ThresholdOptimizer
from tqdm import tqdm
import json

def get_model(config):
    if config['model']['architecture'] == 'graphgps':
        return GraphGPSModel(config)
    elif config['model']['architecture'] == 'graphformer':
        return GraphormerModel(config)
    raise ValueError(f"Unknown architecture: {config['model']['architecture']}")

def get_cf_collate_fn(config):
    base_collator = GraphTransformerCollator(config)
    def collate_fn(batch_list):
        clean_list = [item['clean'] for item in batch_list]
        corrupted_list = [item['corrupted'] for item in batch_list]
        return {
            'clean': base_collator(clean_list),
            'corrupted': base_collator(corrupted_list),
            'clean_list': clean_list
        }
    return collate_fn

def compute_edge_attributions_for_dataloader(engine, dataloader, loss_fn, device):
    """Specifically extracts edge-level attributions for all graphs in a dataloader."""
    all_edge_attrs = []
    for batch in tqdm(dataloader, desc="Edge Scoring", leave=False):
        def to_device(obj, dev):
            if hasattr(obj, 'to'): return obj.to(dev)
            if isinstance(obj, dict): return {k: v.to(dev) if hasattr(v, 'to') else v for k, v in obj.items()}
            return obj
        
        clean_batch = to_device(batch['clean'], device)
        corrupted_batch = to_device(batch['corrupted'], device)
        
        batch_micro = engine.compute_edge_attributions(clean_batch, corrupted_batch, loss_fn)
        
        if isinstance(clean_batch, dict):
            batch_size = len(batch['clean_list'])
        else:
            batch_size = clean_batch.num_graphs if hasattr(clean_batch, 'num_graphs') else 1
            
        for i in range(batch_size):
            graph_micro = {k: v[i].detach().cpu() for k, v in batch_micro.items() if v.dim() == 4}
            all_edge_attrs.append(graph_micro)
    return all_edge_attrs

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

    node_attr_path = os.path.join(save_dir, 'node_attributions.pt')
    node_attributions = None
    if os.path.exists(node_attr_path):
        print(f"Loading node attributions from {node_attr_path}...")
        node_attributions = torch.load(node_attr_path, map_location=device)
    else:
        print(f"Warning: node_attributions.pt not found at {node_attr_path}. Faithfulness evaluation will be skipped.")

    optimizer = ThresholdOptimizer(
        engine, model, val_dataloader, test_dataloader, device, 
        tolerance=args.tolerance, 
        dataset_name=config['dataset']['name'],
        config=config
    )
    
    loss_fn = torch.nn.CrossEntropyLoss()

    # --- TWO-PHASE DISCOVERY PIPELINE ---
    if target_classes:
        print("\n" + "="*50)
        print("STARTING TWO-PHASE CIRCUIT DISCOVERY")
        print("="*50)

        # Phase 1: Topology Only (Graph Circuit)
        print("Computing edge attributions for Validation set (Phase 1)...")
        val_edge_attrs = compute_edge_attributions_for_dataloader(engine, val_dataloader, loss_fn, device)
        best_edge_masks, best_edge_percentile = optimizer.optimize_graph_circuit(val_edge_attrs, save_dir)
        
        # Phase 2: Components Only (Model Circuit)
        print("\n--- Phase 2: Model-Level Circuit Discovery (Components) ---")
        optimizer.optimize(attributions, save_dir, node_attributions=node_attributions)
        
        # Phase 3: Joint Analysis & Testing
        print("\n--- Phase 3: Joint Analysis (Graph + Model Circuit) ---")
        with open(os.path.join(save_dir, 'optimal_masks.pt'), 'rb') as f:
            best_component_masks = torch.load(f, map_location=device)
            
        print("Computing edge attributions for Test set (Phase 3)...")
        test_edge_attrs = compute_edge_attributions_for_dataloader(engine, test_dataloader, loss_fn, device)
        test_edge_masks, _ = optimizer.create_edge_masks_from_percentile(test_edge_attrs, best_edge_percentile)

        # -------------------------------------------------------------
        # Evaluate pure Graph Circuit on the test dataset 
        # (Pass None to component_masks to leave the model brain fully active)
        # -------------------------------------------------------------
        print("Evaluating pure Graph Circuit (Edges Only) on Test Set...")
        graph_circuit_f1 = optimizer.evaluate_patched_model(
            component_masks=None, 
            edge_masks=test_edge_masks, 
            dataloader=test_dataloader
        )
        print(f"Graph Circuit (Edges only) Test F1: {graph_circuit_f1:.4f}")

        # Evaluate the Joint Circuit
        joint_f1 = optimizer.evaluate_patched_model(
            component_masks=best_component_masks, 
            edge_masks=test_edge_masks, 
            dataloader=test_dataloader
        )
        print(f"Joint Circuit (Graph + Model) Test F1: {joint_f1:.4f}")
        
        # -------------------------------------------------------------
        # Update JSON to save all metrics
        # -------------------------------------------------------------
        joint_results = {
            "phase1_graph_percentile": float(best_edge_percentile),
            "graph_circuit_f1": float(graph_circuit_f1),
            "joint_f1": float(joint_f1)
        }
        with open(os.path.join(save_dir, 'joint_discovery_results.json'), 'w') as f:
            json.dump(joint_results, f, indent=4)
    else:
        # Standard optimization if no wrappers
        optimizer.optimize(attributions, save_dir, node_attributions=node_attributions)

    # -----------------------------------------------------------------
    # NEW CAUSAL BENCHMARK EVALUATION: Computes K-Node Induced Subgraphs 
    # and dumps the factual drop in F1 scores to a JSON file.
    # -----------------------------------------------------------------
    print("\n" + "-"*60)
    print("RUNNING TRUE CAUSAL NODE-INDUCED SUBGRAPH BASELINES")
    print("-"*60)
    optimizer.run_node_selection_baselines(save_dir=save_dir, dataloader=test_dataloader)

if __name__ == "__main__":
    main()