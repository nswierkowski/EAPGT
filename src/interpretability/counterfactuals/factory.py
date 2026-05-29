import os
import torch
from tqdm import tqdm

from src.interpretability.counterfactuals.ba_shape import BAShapesCounterfactual
from src.interpretability.counterfactuals.zinc import ZINCCounterfactual
from src.data.transforms.transform_factory import get_transform

def get_counterfactual_engine(config):
    """Returns the engine responsible for mutating a single graph."""
    name = config['dataset']['name'].lower()
    if name == 'ba_shapes':
        return BAShapesCounterfactual(config)
    elif name == 'zinc_no2':
        return ZINCCounterfactual(config)
    else:
        raise ValueError(f"No counterfactual engine for {name}")

def get_counterfactual_dataset(config, base_dataset=None):
    """
    Checks if the counterfactual dataset exists on disk.
    If it does, loads and returns it.
    If not, generates it using the appropriate engine and base dataset.
    """
    root_dir = config['dataset']['root_dir']
    cf_path = os.path.join(root_dir, 'counterfactuals.pt')
    
    if os.path.exists(cf_path):
        print(f"[Counterfactuals] Found existing file at {cf_path}. Loading from disk...")
        return torch.load(cf_path, weights_only=False)
        
    print(f"[Counterfactuals] File not found at {cf_path}. Starting generation...")
    
    if base_dataset is None:
        raise ValueError("A 'base_dataset' must be provided to generate counterfactuals from scratch.")
        
    engine = get_counterfactual_engine(config)
    transform = get_transform(config)
    
    counterfactuals = []
    
    skipped_count = 0
    success_count = 0
    
    print(f"LEN OF base_dataset: {len(base_dataset)}")
    for i in tqdm(range(len(base_dataset)), desc="Generating Pairs"):
        clean_data = base_dataset[i]
        
        # Generate counterfactual (it deepcopies the already-transformed clean_data)
        corrupted_data = engine.generate(clean_data)
        
        # Detect if the example was skipped or failed
        if corrupted_data is clean_data or corrupted_data is None:
            skipped_count += 1
            continue
            
        if transform: 
            corrupted_data = transform(corrupted_data)
            
        counterfactuals.append({
            'index': i,
            'clean': clean_data,
            'corrupted': corrupted_data
        })
        success_count += 1

    print(f"\n[{'='*40}]")
    print(f"[Counterfactuals] Generation Summary:")
    print(f"  Total Processed:  {len(base_dataset)}")
    print(f"  Successfully Generated: {success_count}")
    print(f"  Lost/Skipped:     {skipped_count}")
    print(f"[{'='*40}]\n")

    if success_count == 0:
        print("WARNING: No counterfactuals were generated! Check your generation logic.")
        
    torch.save(counterfactuals, cf_path)
    print(f"Saved {len(counterfactuals)} counterfactual pairs to {cf_path}")
    
    return counterfactuals