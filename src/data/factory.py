import os
import torch
from src.data.ba_shapes.dataset import BAShapesDataset
from src.data.transforms.transform_factory import get_transform
from src.data.zinc.dataset import ZINCNO2Dataset

def get_dataset(config):
    dataset_name = config['dataset']['name'].lower()
    root = config['dataset']['root_dir']
    
    processed_map = {
        'ba_shapes': 'ba_shapes_graph_level.pt',
        'zinc_no2': 'zinc_no2_data.pt'
    }
    
    expected_file = processed_map.get(dataset_name)
    processed_path = os.path.join(root, 'processed', expected_file) if expected_file else None
    
    if processed_path and os.path.exists(processed_path):
        print(f"[Dataset] Found processed {dataset_name} at {processed_path}. Loading from disk...")
    else:
        print(f"[Dataset] {dataset_name} not found or incomplete. Starting generation/processing...")

    pre_transform = get_transform(config)
    
    if dataset_name == 'ba_shapes':
        return BAShapesDataset(
            root=root, 
            config=config, 
            pre_transform=pre_transform  
        )
    elif dataset_name == 'zinc_no2':
        return ZINCNO2Dataset(
            root=root, 
            config=config, 
            pre_transform=pre_transform
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")