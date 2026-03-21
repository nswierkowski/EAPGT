from src.data.transforms.combined_transform import CombinedTransform
from src.data.transforms.graphormer_transform import GraphormerTransform
from src.data.transforms.graphgps_transform import GraphGPSTransform

def get_transform(config):
    transform_name = config['dataset'].get('transform', 'none').lower()
    
    if transform_name == 'combined':
        return CombinedTransform(config)
    elif transform_name == 'graphormer':
        return GraphormerTransform(config.get('transform_params', {}))
    elif transform_name == 'graphgps':
        return GraphGPSTransform(config.get('transform_params', {}))
        
    raise ValueError(f"Unknown transform requested: {transform_name}")