import os
import glob
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
    Checks if the counterfactual dataset exists on disk (legacy single file or chunked).
    If it does, loads and returns it.
    If not, generates it using the appropriate engine and base dataset, saving in chunks to avoid OOM.
    """
    root_dir = config['dataset']['root_dir']
    legacy_cf_path = os.path.join(root_dir, 'counterfactuals.pt')
    cf_dir = os.path.join(root_dir, 'counterfactuals')
    
    # 1. Check for legacy monolithic file
    if os.path.exists(legacy_cf_path):
        print(f"[Counterfactuals] Found legacy file at {legacy_cf_path}. Loading...")
        return torch.load(legacy_cf_path, weights_only=False)
        
    # 2. Check for existing chunks
    os.makedirs(cf_dir, exist_ok=True)
    existing_chunks = sorted(glob.glob(os.path.join(cf_dir, 'cf_chunk_*.pt')))
    
    if existing_chunks:
        print(f"[Counterfactuals] Found {len(existing_chunks)} existing chunks in {cf_dir}. Loading into memory...")
        all_cfs = []
        for chunk_file in tqdm(existing_chunks, desc="Loading Chunks"):
            all_cfs.extend(torch.load(chunk_file, weights_only=False))
        return all_cfs

    # 3. Generate from scratch if neither exists
    print(f"[Counterfactuals] No existing data found. Starting generation...")
    if base_dataset is None:
        raise ValueError("A 'base_dataset' must be provided to generate counterfactuals from scratch.")
        
    engine = get_counterfactual_engine(config)
    transform = get_transform(config)
    
    counterfactuals = []
    skipped_count = 0
    success_count = 0
    chunk_size = 5000  # Safe threshold to avoid OOM. Adjust if your graphs are exceptionally large.
    chunk_idx = 0
    
    print(f"LEN OF base_dataset: {len(base_dataset)}")
    for i in tqdm(range(len(base_dataset)), desc="Generating Pairs"):
        clean_data = base_dataset[i]
        
        # Generate counterfactual 
        corrupted_data = engine.generate(clean_data)
        
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

        # Dump to disk and clear list when chunk is full to prevent OOM
        if len(counterfactuals) >= chunk_size:
            chunk_path = os.path.join(cf_dir, f'cf_chunk_{chunk_idx}.pt')
            torch.save(counterfactuals, chunk_path)
            counterfactuals = []  # Free memory
            chunk_idx += 1

    # Save any remaining items
    if counterfactuals:
        chunk_path = os.path.join(cf_dir, f'cf_chunk_{chunk_idx}.pt')
        torch.save(counterfactuals, chunk_path)

    print(f"\n[{'='*40}]")
    print(f"[Counterfactuals] Generation Summary:")
    print(f"  Total Processed:  {len(base_dataset)}")
    print(f"  Successfully Generated: {success_count}")
    print(f"  Lost/Skipped:     {skipped_count}")
    print(f"[{'='*40}]\n")

    if success_count == 0:
        print("WARNING: No counterfactuals were generated! Check your generation logic.")
        return []
        
    print(f"Saved {success_count} total counterfactual pairs across chunks in {cf_dir}")
    
    # Recursively call to trigger the loading logic (Step 2) and return the full list
    return get_counterfactual_dataset(config, base_dataset=None)