import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, Optional
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import scatter
from src.models.wrappers.base import (
    BaseMPNNWrapper, BaseMessageModule, BaseAggregateModule, BaseUpdateModule
)

class ResGatedMessage(BaseMessageModule):
    """Explicit module for the Message step."""
    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer

    def forward(self, k_i: Tensor, q_j: Tensor, v_j: Tensor, edge_attr: OptTensor = None) -> Tensor:
        if edge_attr is not None:
            k_i = self.base_layer.lin_key(torch.cat([k_i, edge_attr], dim=-1))
            q_j = self.base_layer.lin_query(torch.cat([q_j, edge_attr], dim=-1))
            v_j = self.base_layer.lin_value(torch.cat([v_j, edge_attr], dim=-1))

        return self.base_layer.act(k_i + q_j) * v_j


class ResGatedAggregate(BaseAggregateModule):
    """Explicit module for the Aggregate step."""
    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer
        self.aggr = base_layer.aggr if hasattr(base_layer, 'aggr') else 'add'

    def forward(self, messages: Tensor, index: Tensor, dim_size: int) -> Tensor:
        return scatter(messages, index, dim=0, dim_size=dim_size, reduce=self.aggr)


class ResGatedUpdate(BaseUpdateModule):
    """Explicit module for the Update step (skip connection & bias)."""
    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer

    def forward(self, aggr_out: Tensor, x_target: Tensor) -> Tensor:
        out = aggr_out
        
        if self.base_layer.root_weight:
            out = out + self.base_layer.lin_skip(x_target)

        if self.base_layer.bias is not None:
            out = out + self.base_layer.bias

        return out


class WrappedResGatedGraphConv(BaseMPNNWrapper):
    """
    The main wrapper. Takes a trained PyG ResGatedGraphConv and perfectly 
    replicates its forward pass using trackable nn.Module subcomponents.
    """
    def __init__(self, base_layer: nn.Module):
        super().__init__(
            message_module=ResGatedMessage(base_layer),
            aggregate_module=ResGatedAggregate(base_layer),
            update_module=ResGatedUpdate(base_layer)
        )
        self.base_layer = base_layer

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: OptTensor = None) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        if self.base_layer.edge_dim is None:
            k = self.base_layer.lin_key(x[1])
            q = self.base_layer.lin_query(x[0])
            v = self.base_layer.lin_value(x[0])
        else:
            k, q, v = x[1], x[0], x[0]

        src, dst = edge_index[0], edge_index[1]
        k_i = k[dst]
        q_j = q[src]
        v_j = v[src]

        messages = self.message_module(k_i, q_j, v_j, edge_attr)

        dim_size = x[1].size(0)
        aggr_out = self.aggregate_module(messages, index=dst, dim_size=dim_size)

        out = self.update_module(aggr_out, x_target=x[1])

        return out