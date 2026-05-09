import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.models.wrappers.base import (
    BaseMPNNWrapper, BaseMessageModule, BaseAggregateModule, BaseUpdateModule
)

class AttentionMessage(BaseMessageModule):
    def __init__(self, mha_module: nn.MultiheadAttention):
        super().__init__()
        self.embed_dim = mha_module.embed_dim
        self.num_heads = mha_module.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        
        self.in_proj_weight = mha_module.in_proj_weight
        self.in_proj_bias = mha_module.in_proj_bias

    def forward(self, x):
        B, N, D = x.shape
        
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        return attn_weights, v

class AttentionAggregate(BaseAggregateModule):
    def __init__(self):
        super().__init__()

    def forward(self, attn_weights, v):
        aggregated = torch.matmul(attn_weights, v)
        return aggregated

class AttentionUpdate(BaseUpdateModule):
    def __init__(self, mha_module: nn.MultiheadAttention):
        super().__init__()
        self.num_heads = mha_module.num_heads
        self.out_proj = mha_module.out_proj

    def forward(self, aggregated):
        B, H, N, HD = aggregated.shape
        
        concat_heads = aggregated.permute(0, 2, 1, 3).reshape(B, N, -1)
        
        out = self.out_proj(concat_heads)
        return out

class WrappedAttentionMPNN(nn.MultiheadAttention):
    """
    Wraps standard PyTorch MHA into an MPNN structure.
    Inherits from nn.MultiheadAttention to bypass PyG's strict isinstance checks.
    """
    def __init__(self, original_mha: nn.MultiheadAttention):
        super().__init__(
            embed_dim=original_mha.embed_dim,
            num_heads=original_mha.num_heads,
            dropout=original_mha.dropout,
            bias=original_mha.in_proj_bias is not None,
            add_bias_kv=original_mha.bias_k is not None,
            add_zero_attn=original_mha.add_zero_attn,
            kdim=original_mha.kdim,
            vdim=original_mha.vdim,
            batch_first=original_mha.batch_first
        )
        
        self.load_state_dict(original_mha.state_dict())

        self.message_module = AttentionMessage(self)
        self.aggregate_module = AttentionAggregate()
        self.update_module = AttentionUpdate(self)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
        """
        Matches PyTorch's native MHA signature exactly so PyG doesn't crash, 
        but routes the math through our M, A, U nodes for NNsight tracing!
        """
        # In GraphGPS self-attention, query == key == value == h
        # We extract 'query' to act as our standard graph node features 'x'
        x = query 
        
        is_seq_first = not self.batch_first
        if is_seq_first:
            x = x.transpose(0, 1)

        attn_weights, v = self.message_module(x)
        
        aggregated = self.aggregate_module(attn_weights, v)
        
        out = self.update_module(aggregated)
        
        if is_seq_first:
            out = out.transpose(0, 1)
            
        return out, None