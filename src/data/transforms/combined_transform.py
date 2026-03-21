from src.data.transforms.graphormer_transform import GraphormerTransform
from src.data.transforms.graphgps_transform import GraphGPSTransform

class CombinedTransform:
    """
    Applies both Graphormer (spatial/shortest-path) and GraphGPS (random walk) 
    positional encodings to the same PyG Data object.
    """
    def __init__(self, config):
        t_params = config.get('transform_params', {})
        
        self.graphormer_tf = GraphormerTransform(t_params)
        self.gps_tf = GraphGPSTransform(t_params)

    def __call__(self, data):
        data = self.graphormer_tf(data)        
        data = self.gps_tf(data)
        
        return data