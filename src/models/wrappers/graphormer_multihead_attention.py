import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from src.models.wrappers.base import (
    BaseMPNNWrapper, BaseMessageModule, BaseAggregateModule, BaseUpdateModule
)

class GraphormerAttentionMessage(BaseMessageModule):
    def __init__(self, hf_attn_module: nn.Module):
        super().__init__()
        self.num_heads = hf_attn_module.num_heads
        self.head_dim = hf_attn_module.head_dim
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = hf_attn_module.q_proj
        self.k_proj = hf_attn_module.k_proj
        self.v_proj = hf_attn_module.v_proj

    def forward(self, hidden_states: torch.Tensor, attn_bias: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None):
        
        # 1. Detect if input is [Nodes, Batch, Hidden] (Fairseq/Graphormer style)
        is_seq_first = False
        if attn_bias is not None:
            # attn_bias is always [..., max_nodes, max_nodes], so the last dim is exactly N
            N_expected = attn_bias.size(-1)
            if hidden_states.size(0) == N_expected:
                is_seq_first = True
                N, B, C = hidden_states.shape
            else:
                B, N, C = hidden_states.shape
        else:
            # Fallback assumption for Graphormer
            N, B, C = hidden_states.shape
            is_seq_first = True

        # 2. Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Transpose to standard [Batch, Nodes, Hidden] for EAP math
        if is_seq_first:
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
            
        # 3. Reshape to [Batch, Heads, Nodes, Head_Dim]
        q = q.contiguous().view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.contiguous().view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.contiguous().view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 4. Core Message Score (Similarity)
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        
        # 5. INJECT GRAPH INDUCTIVE BIASES
        if attn_bias is not None:
            if attn_bias.dim() == 3:
                # attn_bias usually arrives as [B * Heads, N, N]
                attn_bias = attn_bias.contiguous().view(B, self.num_heads, N, N)
            attn_weights = attn_weights + attn_bias
            
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # 6. Final Message weights
        attn_probs = F.softmax(attn_weights, dim=-1)
        

        if attn_probs.requires_grad:
            attn_probs.retain_grad()
        if v.requires_grad:
            v.retain_grad()

        return attn_probs, v, is_seq_first


class GraphormerAttentionAggregate(BaseAggregateModule):
    def __init__(self):
        super().__init__()

    def forward(self, attn_probs: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Aggregate step: A_{ij} * V_j
        attn_output = torch.matmul(attn_probs, v)
        
        if attn_output.requires_grad:
            attn_output.retain_grad()
            
        return attn_output


class GraphormerAttentionUpdate(BaseUpdateModule):
    def __init__(self, hf_attn_module: nn.Module):
        super().__init__()
        self.out_proj = hf_attn_module.out_proj

    def forward(self, attn_output: torch.Tensor, is_seq_first: bool) -> torch.Tensor:
        B, Heads, N, Head_Dim = attn_output.shape
        
        # Reshape back to [Batch, Nodes, Hidden_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, N, Heads * Head_Dim)
        
        # Final update projection
        attn_output = self.out_proj(attn_output)
        
        # Revert back to [Nodes, Batch, Hidden_Dim] if the parent model expects it
        if is_seq_first:
            attn_output = attn_output.transpose(0, 1).contiguous()
            
        if attn_output.requires_grad:
            attn_output.retain_grad()
            
        return attn_output


class WrappedGraphormerMultiheadAttention(BaseMPNNWrapper):
    """
    Wraps the Hugging Face GraphormerMultiheadAttention module.
    """
    def __init__(self, base_layer: nn.Module):
        super().__init__(
            message_module=GraphormerAttentionMessage(base_layer),
            aggregate_module=GraphormerAttentionAggregate(),
            update_module=GraphormerAttentionUpdate(base_layer)
        )
        self.base_layer = base_layer

    def forward(
        self, 
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None, 
        attn_bias: Optional[torch.Tensor] = None, 
        query: Optional[torch.Tensor] = None,
        **kwargs
    ):
        if hidden_states is None:
            hidden_states = query
            
        attn_probs, v, is_seq_first = self.message_module(hidden_states, attn_bias=attn_bias, attention_mask=attention_mask)
        
        attn_output_aggregated = self.aggregate_module(attn_probs, v)
        
        output = self.update_module(attn_output_aggregated, is_seq_first)

        return output, attn_probs