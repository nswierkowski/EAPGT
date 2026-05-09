import torch
import torch.nn as nn
import torch.nn.functional as F

class WrappedGraphormerGraphAttnBias(nn.Module):
    def __init__(self, original_module: nn.Module):
        super().__init__()
        self.original = original_module
        self.num_heads = original_module.num_heads
        
        
        self.spatial_tracker = nn.Identity()
        self.edge_tracker = nn.Identity()
        self.combined_tracker = nn.Identity()

    def forward(self, input_nodes, attn_bias, spatial_pos, input_edges, attn_edge_type):
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        
        max_dist = self.original.spatial_pos_encoder.weight.shape[0] - 1
        safe_pos = torch.where(spatial_pos < 0, max_dist, spatial_pos)
        safe_pos = torch.clamp(safe_pos, min=0, max=max_dist)
        
        spatial_bias = self.original.spatial_pos_encoder(safe_pos).permute(0, 3, 1, 2)
        
        spatial_bias = self.spatial_tracker(spatial_bias)
        graph_attn_bias = graph_attn_bias + F.pad(spatial_bias, (1, 0, 1, 0), value=0.0)

        max_edges = self.original.edge_encoder.weight.shape[0] - 1
        safe_edges = torch.where(input_edges < 0, max_edges, input_edges)
        safe_edges = torch.clamp(safe_edges, min=0, max=max_edges)
        
        edge_feat = self.original.edge_encoder(safe_edges)
        
        if getattr(self.original, 'edge_dis_encoder', None) is not None and attn_edge_type is not None:
            max_dis = self.original.edge_dis_encoder.weight.shape[0] - 1
            safe_attn = torch.where(attn_edge_type < 0, max_dis, attn_edge_type)
            safe_attn = torch.clamp(safe_attn, min=0, max=max_dis)
            
            attn_feat = self.original.edge_dis_encoder(safe_attn)
            edge_feat = edge_feat + attn_feat.unsqueeze(-2)
        
        edge_bias = edge_feat.sum(dim=3).squeeze(-2).permute(0, 3, 1, 2)
        
        edge_bias = self.edge_tracker(edge_bias)
        graph_attn_bias = graph_attn_bias + F.pad(edge_bias, (1, 0, 1, 0), value=0.0)

        graph_attn_bias = self.combined_tracker(graph_attn_bias)
            
        return graph_attn_bias