from src.data.ba_shapes.dataset import BAShapesDataset
from src.data.transforms.transform_factory import get_transform
from src.data.zinc.dataset import ZINCNO2Dataset

def get_dataset(config):
    dataset_name = config['dataset']['name'].lower()
    root = config['dataset']['root_dir']
    
    pre_transform = get_transform(config)
    
    if dataset_name == 'ba_shapes':
        return BAShapesDataset(
            root=root, 
            config=config['dataset'], 
            pre_transform=pre_transform  
        )
    elif dataset_name == 'zinc_no2':
        return ZINCNO2Dataset(root=root, config=config['dataset'], pre_transform=pre_transform)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")