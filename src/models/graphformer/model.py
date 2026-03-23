import torch
import torch.nn as nn
from transformers import GraphormerForGraphClassification, GraphormerConfig
from src.models.base import BaseGraphTransformer

class ContinuousFeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_size)
        
    def forward(self, x):
        out = self.proj(x) 
        return out.unsqueeze(-2) 

class GraphormerModel(BaseGraphTransformer):
    def __init__(self, config):
        super().__init__(config)
        
        model_config = config['model']
        dataset_config = config.get('dataset', {})
        
        dataset_name = dataset_config.get('name', 'ba_shapes') 
        
        pretrained_path = model_config.get('pretrained_path', None)
        num_classes = model_config.get('num_classes', 2)
        input_dim = model_config.get('input_dim', 10)
        
        hidden_size = model_config.get('hidden_dim', 768)
        num_layers = model_config.get('num_layers', 12)
        num_heads = model_config.get('num_heads', 32)
        
        if pretrained_path:
            print(f"Loading pre-trained Graphormer strictly from: {pretrained_path}")
            print("WARNING: Custom architecture sizes (hidden_dim, layers) are ignored when loading pre-trained weights.")
            hf_config = GraphormerConfig.from_pretrained(pretrained_path)
            self.hf_model = GraphormerForGraphClassification.from_pretrained(
                pretrained_path, config=hf_config
            )
            
            if hf_config.num_labels != num_classes:
                print(f"Replacing classification head: {hf_config.num_labels} -> {num_classes} classes")
                self.hf_model.config.num_labels = num_classes
                if hasattr(self.hf_model.classifier, 'out_proj'):
                    self.hf_model.classifier.out_proj = nn.Linear(self.hf_model.classifier.out_proj.in_features, num_classes)
                else:
                    self.hf_model.classifier = nn.Linear(self.hf_model.config.hidden_size, num_classes)
        else:
            print(f"Initializing untrained Graphormer from scratch (Layers: {num_layers}, Dim: {hidden_size}, Heads: {num_heads}).")
            hf_config = GraphormerConfig(
                num_labels=num_classes,
                hidden_size=hidden_size,
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads
            )
            self.hf_model = GraphormerForGraphClassification(hf_config)

        if dataset_name == 'ba_shapes':
            print("Detected BA-Shapes: Applying ContinuousFeatureEncoder patch.")
            actual_hidden_size = self.hf_model.config.hidden_size
            self.hf_model.encoder.graph_encoder.graph_node_feature.atom_encoder = ContinuousFeatureEncoder(input_dim, actual_hidden_size)
        else:
            print(f"Detected {dataset_name}: Keeping Hugging Face's discrete atom encoder for chemistry.")
            
    def forward(self, batch):
        hf_inputs = {
            'input_nodes': batch.get('input_nodes'),
            'attn_bias': batch.get('attn_bias'),
            'in_degree': batch.get('in_degree'),
            'out_degree': batch.get('out_degree'),
            'spatial_pos': batch.get('spatial_pos'),
        }
        
        device = hf_inputs['input_nodes'].device
        bsz, seq_len = hf_inputs['input_nodes'].shape[:2]
        max_dist = self.hf_model.config.multi_hop_max_dist 
        
        if 'input_edges' in batch:
            hf_inputs['input_edges'] = batch['input_edges']
        else:
            hf_inputs['input_edges'] = torch.ones(
                (bsz, seq_len, seq_len, max_dist, 1), 
                dtype=torch.long, 
                device=device
            )

        if 'attn_edge_type' in batch:
            hf_inputs['attn_edge_type'] = batch['attn_edge_type']
        else:
            hf_inputs['attn_edge_type'] = torch.ones(
                (bsz, seq_len, seq_len, max_dist), 
                dtype=torch.long, 
                device=device
            )

        outputs = self.hf_model(**hf_inputs)
        return outputs.logits
        
    def get_patchable_components(self):
        layers = self.hf_model.encoder.graph_encoder.layers
        return {
            "self_attentions": [layer.self_attn for layer in layers],
            "mlps": [layer.fc1 for layer in layers]
        }