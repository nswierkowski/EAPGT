import os
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
        checkpoint_path = model_config.get('checkpoint_path', None)
        num_classes = model_config.get('num_classes', 2)
        input_dim = model_config.get('input_dim', 10)
        
        hidden_size = model_config.get('hidden_dim', 768)
        
        print(f"Model Config: hidden_dim={hidden_size}, num_layers={model_config.get('num_layers', 'default')}, num_heads={model_config.get('num_heads', 'default')}")
        num_layers = model_config.get('num_layers', 12)
        num_heads = model_config.get('num_heads', 32)
        
        if pretrained_path:
            print(f"Loading pre-trained Graphormer strictly from: {pretrained_path}")
            print("WARNING: Custom architecture sizes (hidden_dim, layers) are ignored when loading pre-trained weights.")
            hf_config = GraphormerConfig.from_pretrained(pretrained_path)
            self.hf_model = GraphormerForGraphClassification.from_pretrained(
                pretrained_path, config=hf_config
            )
            
            if checkpoint_path:
                self._load_checkpoint(checkpoint_path)
            
            if hf_config.num_labels != num_classes:
                print(f"Replacing classification head: {hf_config.num_labels} -> {num_classes} classes")
                self.hf_model.config.num_labels = num_classes
                if hasattr(self.hf_model.classifier, 'out_proj'):
                    self.hf_model.classifier.out_proj = nn.Linear(self.hf_model.classifier.out_proj.in_features, num_classes)
                else:
                    self.hf_model.classifier = nn.Linear(self.hf_model.config.hidden_size, num_classes)
        else:
            #raise RuntimeError("Should read model xdd")
            print(f"Initializing untrained Graphormer from scratch (Layers: {num_layers}, Dim: {hidden_size}, Heads: {num_heads}).")
            hf_config = GraphormerConfig(
                num_labels=num_classes,
                hidden_size=hidden_size,
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads
            )
            self.hf_model = GraphormerForGraphClassification(hf_config)
            
            if hf_config.num_labels != num_classes:
                print(f"Replacing classification head: {hf_config.num_labels} -> {num_classes} classes")
                self.hf_model.config.num_labels = num_classes
                if hasattr(self.hf_model.classifier, 'out_proj'):
                    self.hf_model.classifier.out_proj = nn.Linear(self.hf_model.classifier.out_proj.in_features, num_classes)
                else:
                    self.hf_model.classifier = nn.Linear(self.hf_model.config.hidden_size, num_classes)

        if dataset_name == 'ba_shapes':
            print("Detected BA-Shapes: Applying ContinuousFeatureEncoder patch.")
            actual_hidden_size = self.hf_model.config.hidden_size
            self.hf_model.encoder.graph_encoder.graph_node_feature.atom_encoder = ContinuousFeatureEncoder(input_dim, actual_hidden_size)
        else:
            print(f"Detected {dataset_name}: Keeping Hugging Face's discrete atom encoder for chemistry.")
            
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
            
    def load_state_dict(self, state_dict, strict=False):
        """
        Intercept external calls (e.g., from run_eap.py) to safely remap prefixes 
        to match the self.hf_model wrapper format.
        """
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'model_state' in state_dict:
            state_dict = state_dict['model_state']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        mapped_dict = {}
        for k, v in state_dict.items():
            if k in ['epoch', 'optimizer_state_dict']:
                continue
                
            new_key = k
            new_key = new_key.replace('module.', '').replace('model.', '').replace('net.', '')
            
            if new_key.startswith('hf_encoder.'):
                new_key = new_key.replace('hf_encoder.', 'hf_model.encoder.')
            elif new_key.startswith('hf_classifier.'):
                new_key = new_key.replace('hf_classifier.', 'hf_model.classifier.')
                
            mapped_dict[new_key] = v
            
        return super().load_state_dict(mapped_dict, strict=strict)

    def _load_checkpoint(self, path):
        """Internal initialization loader with strict shape and key checking."""
        print(f"Loading Graphormer checkpoint from {path}...")
        if not os.path.exists(path):
            print(f"WARNING: Checkpoint {path} not found. Training from scratch.")
            return

        checkpoint = torch.load(path, map_location='cpu')
        
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint: 
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        cleaned_dict = {}
        for k, v in state_dict.items():
            new_key = k
            
            new_key = new_key.replace('module.', '').replace('model.', '').replace('net.', '')
            new_key = new_key.replace('hf_model.', '')
            
            if new_key.startswith('hf_encoder.'):
                new_key = new_key.replace('hf_encoder.', 'encoder.')
            elif new_key.startswith('hf_classifier.'):
                new_key = new_key.replace('hf_classifier.', 'classifier.')
                
            cleaned_dict[new_key] = v

        current_model_dict = self.hf_model.state_dict()
        keys_to_delete = []
        
        for k in list(cleaned_dict.keys()):
            if k not in current_model_dict:
                print(f"Discarding unexpected key: {k}")
                keys_to_delete.append(k)
                continue
                
            if cleaned_dict[k].shape != current_model_dict[k].shape:
                print(f"Shape mismatch for {k}: checkpoint {tuple(cleaned_dict[k].shape)} vs model {tuple(current_model_dict[k].shape)}. Reinitializing.")
                keys_to_delete.append(k)
                continue
                
            # Always reinitialize the classifier head to gracefully handle num_classes changes
            # if 'classifier' in k:
            #     keys_to_delete.append(k)
                
        for k in keys_to_delete:
            del cleaned_dict[k]
            
        # Merge into current state dict (keeps random init for missing keys, strict shape check for present keys)
        current_model_dict.update(cleaned_dict)
        self.hf_model.load_state_dict(current_model_dict, strict=True)
        print(f"Successfully loaded {len(cleaned_dict)} parameters from checkpoint.")

    def forward(self, batch):
        hf_inputs = {
            'input_nodes': batch['input_nodes'],
            'attn_bias': batch['attn_bias'],      
            'in_degree': batch['in_degree'],
            'out_degree': batch['out_degree'],
            'spatial_pos': batch['spatial_pos'],
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
        encoder = self.hf_model.encoder.graph_encoder
        components = {
            "minar_node_encoder": encoder.graph_node_feature,
            "minar_edge_attn_bias": encoder.graph_attn_bias, 
            
            "classic_attentions": [layer.self_attn for layer in encoder.layers],
            "classic_mlps": [layer.fc1 for layer in encoder.layers]
        }
        return components