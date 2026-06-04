from networkx import config
import os
import yaml
from src.models.wrapper import instrument_model
import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader as PyG_DataLoader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pickle

from src.models.graphgps.model import GraphGPSModel 
from src.models.graphformer.model import GraphormerModel
from src.interpretability.eap.strategies import MacroMinarEAP
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
    elif config['model']['architecture'] == 'graphformer':
        return GraphormerModel(config)
    else:
        raise ValueError(f"Unknown architecture: {config['model']['architecture']}")

def get_eap_engine(strategy_str: str, model: nn.Module, config: dict):
    strategies = {
        'macro_minar': MacroMinarEAP
    }
    return strategies[strategy_str](model, config)

def compute_global_scores(engine, dataloader, loss_fn, device) -> tuple:
    """
    Aggregates EAP scores across the entire dataset.
    Splits processing into Macro (Component) and Micro (Structural) flows.
    """
    print("Computing global attribution scores...")
    global_macro = {}
    all_micro_edges = []
    total_graphs = 0
    
    for batch in tqdm(dataloader, desc="EAP Scoring"):
        def to_device(obj, dev):
            if hasattr(obj, 'to'):
                return obj.to(dev)
            elif isinstance(obj, dict):
                return {k: v.to(dev) if hasattr(v, 'to') else v for k, v in obj.items()}
            return obj
        
        clean_batch = to_device(batch['clean'], device)
        corrupted_batch = to_device(batch['corrupted'], device)
        
        batch_scores = engine.evaluate_pair(clean_batch, corrupted_batch, loss_fn)
        
        macro_scores = batch_scores['macro']
        for name, score in macro_scores.items():
            score_val = score.item() if isinstance(score, torch.Tensor) else score
            
            if name not in global_macro:
                global_macro[name] = 0.0

            global_macro[name] += score_val
            
        # 2. Micro Scores (UNBATCHING LOGIC)
        micro_scores = batch_scores['micro']
        
        if isinstance(clean_batch, dict):
            individual_graphs = batch['clean_list'] 
            total_batch_edges = 0 
            batch_size = len(individual_graphs)
        else:
            individual_graphs = clean_batch.to_data_list()
            total_batch_edges = clean_batch.num_edges
            batch_size = clean_batch.num_graphs if hasattr(clean_batch, 'num_graphs') else 1
        
        edge_offset = 0
        for i, graph in enumerate(individual_graphs):
            num_edges = graph.num_edges
            
            graph_micro_scores = {}
            for k, v in micro_scores.items():
                if v.dim() == 4: 
                    graph_attn_slice = v[i] 
                    for h_idx in range(graph_attn_slice.shape[0]):
                        head_key = f"{k}.head_{h_idx}"
                        graph_micro_scores[head_key] = graph_attn_slice[h_idx].detach().cpu()
                
                elif total_batch_edges > 0 and v.shape[0] == total_batch_edges:
                    graph_micro_scores[k] = v[edge_offset : edge_offset + num_edges].detach().cpu()
                
                else:
                    graph_micro_scores[k] = v.detach().cpu()
            
            all_micro_edges.append({
                'graph_index': total_graphs + i,
                'edge_index': graph.edge_index.cpu(),
                'micro_scores': graph_micro_scores,
                
                'x': graph.x.cpu() if hasattr(graph, 'x') and graph.x is not None else None,
                'y': graph.y.cpu() if hasattr(graph, 'y') and graph.y is not None else None
            })
            
            edge_offset += num_edges
            
        total_graphs += batch_size

    for name in global_macro.keys():
        global_macro[name] /= total_graphs
        global_macro[name] = torch.tensor(global_macro[name])
        
    return global_macro, all_micro_edges

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

def analyze_attribution_scores(global_macro_scores, save_dir, strategy_name):
    """Saves raw EAP macro scores and plots their distribution."""
    print(f"Saving and analyzing {strategy_name.upper()} macro attribution scores...")

    torch.save(global_macro_scores, os.path.join(save_dir, 'global_macro_attributions.pt'))

    all_scores = []
    for name, score_tensor in global_macro_scores.items():
        all_scores.append(score_tensor.detach().cpu().flatten().abs())
        
    if not all_scores:
        print("Warning: No attribution scores found to plot.")
        return

    flat_scores = torch.cat(all_scores).numpy()

    plt.figure(figsize=(10, 6))
    
    sns.histplot(flat_scores, bins=100, log_scale=(False, True), color='#9b59b6', 
                     label=f"{strategy_name.capitalize()} Scores\nMax: {flat_scores.max():.4f}\nMean: {flat_scores.mean():.6f}")
    plt.title(f"Macro-EAP Component Score Distribution ({strategy_name.capitalize()})", fontsize=14)
    plt.xlabel("Absolute Attribution Score |Act_diff * Grad|")
    plt.ylabel("Count (Log Scale)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(save_dir, 'macro_attribution_distribution.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Saved attribution data to {save_dir}/global_macro_attributions.[pt|png]")
    
def extract_and_plot_circuit(global_scores: dict, graph_states: dict, save_dir: str, threshold_percentile: float = 90.0) -> nx.DiGraph:
    """
    Extracts the macroscopic computational circuit using NetworkX.
    Maps execution flow topologically.
    """
    print(f"Extracting Macroscopic Circuit (Threshold: {threshold_percentile}th percentile)...")
    
    all_scores = torch.cat([score.flatten() for score in global_scores.values()])
    threshold_val = torch.quantile(all_scores.float(), threshold_percentile / 100.0).item()
    
    G = nx.DiGraph()
    ordered_components = list(global_scores.keys())
    
    for comp in ordered_components:
        score_max = global_scores[comp].max().item()
        if score_max >= threshold_val:
            G.add_node(comp, eap_importance=score_max)

    for i in range(len(ordered_components) - 1):
        src = ordered_components[i]
        dst = ordered_components[i+1]
        
        if ".head_" in src and ".head_" in dst:
            src_parent = src.split(".head_")[0]
            dst_parent = dst.split(".head_")[0]
            if src_parent == dst_parent:
                continue 
                
        if ".head_" in src and dst.endswith(".A"):
             if G.has_node(src) and G.has_node(dst):
                 edge_weight = global_scores[dst].max().item()
                 G.add_edge(src, dst, weight=edge_weight)
             continue
        
        if G.has_node(src) and G.has_node(dst):
            edge_weight = global_scores[dst].max().item()
            G.add_edge(src, dst, weight=edge_weight)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    node_sizes = [3000 * (nx.get_node_attributes(G, 'eap_importance')[node]) for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, edge_color='gray', alpha=0.5)
    
    plt.title(f"Macro-MINAR Circuit (Top {100 - threshold_percentile}%)")
    plt.axis("off")
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, "macro_circuit.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    graph_path = os.path.join(save_dir, "macro_circuit.pkl")
    with open(graph_path, "wb") as f:
        pickle.dump(G, f)
    
    print(f"Circuit graph generated with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print(f"Saved graph object to: {graph_path}")
    
    return G
    
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
    
    train_cf_list = [
        cf_pair for cf_pair in cf_data_list
        if hasattr(cf_pair['clean'], 'split_mask') and cf_pair['clean'].split_mask.item() == 0
    ]
    
    cf_collate_fn = get_cf_collate_fn(config)
    dataloader_to_use = DataLoader # if config['model']['architecture'] == 'graphformer' else PyG_DataLoader
    dataloader = dataloader_to_use(
        train_cf_list,  
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
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    target_classes = config['model'].get('target_wrapper_classes', [])
    model = instrument_model(model, target_classes)
    model.eval()
    
    loss_fn = nn.CrossEntropyLoss()
    engine = get_eap_engine(config['eap']['strategy'], model, config)


    global_macro, micro_edges = compute_global_scores(engine, dataloader, loss_fn, device)

    def compute_node_attributions(micro_edges: list) -> list:
        """
        Aggregates edge‑wise micro scores into per‑node scores.
        Correctly aligns attributions to follow target/receiver node conventions.
        """
        node_attributions = []
        for edge_data in micro_edges:
            edge_index = edge_data['edge_index']  # shape (2, E)
            if edge_index.numel() == 0:
                continue
                
            # FIX 1: PyG edge_index[1] represents target/receiver nodes
            target_nodes = edge_index[1] 
            num_edges = target_nodes.shape[0]
            num_nodes = int(target_nodes.max().item()) + 1 if num_edges > 0 else 0
            node_scores = {}
            
            for name, scores in edge_data['micro_scores'].items():
                score_len = scores.shape[-1] 
                
                # FIX 2: Sum over columns (dim=1) to aggregate attention into target rows
                if scores.dim() == 2 and name.endswith(".M"):
                    node_scores[name] = scores.sum(dim=1)
                
                # Node-level scores (already aggregated or from MLPs)
                elif score_len == num_nodes or score_len == num_nodes + 1:
                    node_scores[name] = scores

                # Edge-level scores (Standard MPNN layers)
                elif score_len == num_edges:
                    if scores.dim() == 2:
                        # Head-wise edge scores [heads, num_edges]
                        heads = scores.shape[0]
                        agg = torch.zeros((heads, num_nodes), dtype=scores.dtype, device=scores.device)
                        for i in range(num_edges):
                            tgt = int(target_nodes[i].item())
                            agg[:, tgt] += scores[:, i]
                        node_scores[name] = agg
                    else:
                        # Scalar edge scores [num_edges]
                        agg = torch.zeros((num_nodes,), dtype=scores.dtype, device=scores.device)
                        for i in range(num_edges):
                            tgt = int(target_nodes[i].item())
                            agg[tgt] += scores[i]
                        node_scores[name] = agg
                
                else:
                    node_scores[name] = scores
                    
            # Keep original edge data and append updated node scores
            new_entry = dict(edge_data)
            new_entry['node_scores'] = node_scores
            node_attributions.append(new_entry)
            
        return node_attributions
    
    global_save_path = os.path.join(config['experiment']['save_dir'], 'global_attributions.pt')
    torch.save(global_macro, global_save_path)
    print(f"Saved global attributions to {global_save_path}")

    micro_save_path = os.path.join(config['experiment']['save_dir'], 'micro_edges_raw.pt')
    torch.save(micro_edges, micro_save_path)
    print(f"Saved raw Micro-EAP graph edges and scores to {micro_save_path}")

    # Compute node-level EAP attributions and persist them
    node_attributions = compute_node_attributions(micro_edges)
    node_save_path = os.path.join(config['experiment']['save_dir'], 'node_attributions.pt')
    torch.save(node_attributions, node_save_path)
    print(f"Saved node-level EAP attributions to {node_save_path}")

    analyze_attribution_scores(global_macro, config['experiment']['save_dir'], config['eap']['strategy'])

    circuit_threshold = config['eap'].get('circuit_threshold_percentile', 90.0)
    
    circuit_graph = extract_and_plot_circuit(
        global_scores=global_macro, 
        graph_states=engine.graph_states, 
        save_dir=config['experiment']['save_dir'],  
        threshold_percentile=circuit_threshold
    )


if __name__ == "__main__":
    main()