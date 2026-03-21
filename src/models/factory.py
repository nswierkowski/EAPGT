from src.models.graphformer.model import GraphormerModel
from src.models.graphgps.model import GraphGPSModel 

def get_model(config):
    model_name = config['model']['name'].lower()
    
    if model_name == 'graphormer':
        return GraphormerModel(config)
    elif model_name == 'graphgps':
        return GraphGPSModel(config)
    else:
        raise ValueError(f"Unknown model name: {model_name}")