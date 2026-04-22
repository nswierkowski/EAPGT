import os
import torch
import torch.nn as nn
from torch_geometric.nn import GPSConv, ResGatedGraphConv, global_mean_pool
from src.models.base import BaseGraphTransformer
from src.models.graphgps.layers import FeatureEncoder

class GraphGPSModel(BaseGraphTransformer):
    def __init__(self, config):
        super().__init__(config)
        
        model_config = config['model']
        self.hidden_dim = model_config['hidden_dim']
        num_layers = model_config['num_layers']
        num_heads = model_config['num_heads']
        num_classes = model_config['num_classes']
        dropout = model_config.get('dropout', 0.1)
        
        self.encoder = FeatureEncoder(config)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            local_mpnn = ResGatedGraphConv(
                in_channels=self.hidden_dim, 
                out_channels=self.hidden_dim, 
                edge_dim=self.hidden_dim
            )
            
            conv = GPSConv(
                self.hidden_dim, 
                conv=local_mpnn, 
                heads=num_heads, 
                dropout=dropout#, 
                # attn_dropout=dropout
            )
            self.layers.append(conv)
            
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, num_classes)
        )

        pretrained_path = model_config.get('pretrained_path', None)
        if pretrained_path and pretrained_path.endswith('.ckpt'):
            self._load_checkpoint(pretrained_path)
            

    def _load_checkpoint(self, path):
        print(f"Loading GraphGPS checkpoint from {path}...")
        if not os.path.exists(path):
            print(f"WARNING: Checkpoint {path} not found. Training from scratch.")
            return

        checkpoint = torch.load(path, map_location='cpu')
        
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        cleaned_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('model.', '').replace('net.', '')
            cleaned_dict[new_key] = v

        keys_to_delete = []
        for k in cleaned_dict.keys():
            if 'node_encoder' in k or 'edge_encoder' in k:
                keys_to_delete.append(k)
            elif 'head' in k or 'classifier' in k or 'post_mp' in k:
                keys_to_delete.append(k)
            elif 'pe_encoder' in k and cleaned_dict[k].shape[0] != self.hidden_dim:
                keys_to_delete.append(k)
                
        for k in keys_to_delete:
            del cleaned_dict[k]
            
        print(f"Discarded {len(keys_to_delete)} dataset-specific keys.")

        translated_dict = {}
        discarded_local_count = 0
        
        custom_mpnn_prefixes = [
            'local_A', 'local_B', 'local_C', 'local_D', 'local_E', 
            'local_bn_node_x', 'local_bn_edge_e'
        ]
        
        for k, v in cleaned_dict.items():
            if any(prefix in k for prefix in custom_mpnn_prefixes):
                discarded_local_count += 1
                continue
                
            new_key = k
            
            new_key = new_key.replace('self_attn', 'attn')
            new_key = new_key.replace('ff_linear1', 'mlp.0')
            new_key = new_key.replace('ff_linear2', 'mlp.3')
            
            if 'norm1_local' in new_key:
                new_key = new_key.replace('norm1_local', 'norm1.module')
            elif 'norm1_attn' in new_key:
                new_key = new_key.replace('norm1_attn', 'norm2.module')
            elif 'norm2' in new_key:
                new_key = new_key.replace('norm2', 'norm3.module')
            
            translated_dict[new_key] = v
            
        print(f"Discarded {discarded_local_count} mathematically incompatible local MPNN matrices.")

        current_model_dict = self.state_dict()
        current_model_dict.update(translated_dict)
        
        self.load_state_dict(current_model_dict, strict=True)
        print("Successfully loaded GPS Transformer backbone weights strictly!")

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        
        pe = getattr(batch, 'pe', None)
        edge_attr = getattr(batch, 'edge_attr', None)

        x, edge_attr = self.encoder(x, edge_attr, pe, edge_index)

        for layer in self.layers:
            x = layer(x, edge_index, batch_idx, edge_attr=edge_attr)

        hg = global_mean_pool(x, batch_idx)

        out = self.head(hg)
        return out

    def get_patchable_components(self):
        components = {
            "minar_pe_encoder": getattr(self.encoder, 'pe_encoder', None),
            "minar_edge_encoder": getattr(self.encoder, 'edge_encoder', None),
        }
        
        for i, layer in enumerate(self.layers):
            components[f'minar_layer_{i}_local_mpnn'] = layer.conv            
            
            components[f'classic_layer_{i}_global_attn'] = layer.attn
            
            if hasattr(layer, 'mlp') and isinstance(layer.mlp, nn.Sequential):
                components[f'classic_layer_{i}_mlp_linear1'] = layer.mlp[0]  
                components[f'classic_layer_{i}_mlp_linear2'] = layer.mlp[3]  
            else:
                components[f'classic_layer_{i}_mlp'] = layer.mlp
                
        if hasattr(self, 'head') and isinstance(self.head, nn.Sequential):
             components['classic_head_linear1'] = self.head[0]
             components['classic_head_linear2'] = self.head[3]
            
        return {k: v for k, v in components.items() if v is not None}