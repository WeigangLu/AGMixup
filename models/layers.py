import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GCN2Conv, APPNP, Sequential
from torch.nn import ReLU
from torch_geometric.nn.conv.gcn_conv import gcn_norm

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch.nn import Parameter
import numpy as np

from torch_sparse import SparseTensor, matmul


class Layer(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, layer_type: str, layer_index, bias=True):
        super(Layer, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.layer_type = layer_type
        self.layer_index = layer_index
        self.bias = bias

        self.conv = None
        self.cache = None
    
    def get_hid(self):
        return self.cache
    
    def forward(self, x, edge_index, x_0=None):
        if self.conv is None:
            raise NotImplementedError
        
        if edge_index is not None:
            if x_0 is not None:
                # for GCNII
                out = self.conv(x, x_0, edge_index)
            else:
                out = self.conv(x, edge_index)
        else:
            out = self.conv(x)

        self.cache = out

        return out

class MLPLayer(Layer):
    def __init__(self, in_channels, hid_channels, out_channels, layer_type, layer_index, bias=True):
        super(MLPLayer, self).__init__(in_channels, hid_channels, out_channels, layer_type, layer_index, bias)
        
        if self.layer_type == "in":
            self.conv = nn.Linear(self.in_channels, self.hid_channels, bias=self.bias)
        elif self.layer_type == "hid":
            self.conv = nn.Linear(self.hid_channels, self.hid_channels, bias=self.bias)
        elif self.layer_type == "out":
            self.conv = nn.Linear(self.hid_channels, self.out_channels, bias=self.bias)

class GCNLayer(Layer):
    def __init__(self, in_channels, hid_channels, out_channels, layer_type, layer_index, bias=True):
        super(GCNLayer, self).__init__(in_channels, hid_channels, out_channels, layer_type, layer_index, bias)

        cached = False
        if self.layer_type == "in":
            self.conv = GCNConv(self.in_channels, self.hid_channels, bias=self.bias, cached=cached)
        elif self.layer_type == "hid":
            self.conv = GCNConv(self.hid_channels, self.hid_channels, bias=self.bias, cached=cached)
        elif self.layer_type == "out":
            self.conv = GCNConv(self.hid_channels, self.out_channels, bias=self.bias, cached=cached)


class GATLayer(Layer):
    def __init__(self, in_channels, hid_channels, out_channels, layer_type, layer_index, bias=True, heads=8):
        super(GATLayer, self).__init__(in_channels, hid_channels, out_channels, layer_type, layer_index, bias)

        if self.layer_type == "in":
            self.conv = GATConv(self.in_channels, self.hid_channels, heads=heads, bias=self.bias)
        elif self.layer_type == "hid":
            self.conv = GATConv(self.hid_channels * heads, self.hid_channels, heads=heads, bias=self.bias)
        elif self.layer_type == "out":
            self.conv = GATConv(self.hid_channels * heads, self.out_channels, heads=1, bias=self.bias)


class SAGELayer(Layer):
    def __init__(self, in_channels, hid_channels, out_channels, layer_type, layer_index, bias=True):
        super(SAGELayer, self).__init__(in_channels, hid_channels, out_channels, layer_type, layer_index,
                                        bias)

        if self.layer_type == "in":
            self.conv = SAGEConv(self.in_channels, self.hid_channels, bias=self.bias)
        elif self.layer_type == "hid":
            self.conv = SAGEConv(self.hid_channels, self.hid_channels, bias=self.bias)
        elif self.layer_type == "out":
            self.conv = SAGEConv(self.hid_channels, self.out_channels, bias=self.bias)


class APPNPLayer(Layer):
    def __init__(self, in_channels=1, hid_channels=1, out_channels=1, layer_type="in", layer_index=2,
                 bias=True):
        super(APPNPLayer, self).__init__(in_channels, hid_channels, out_channels, layer_type, layer_index,
                                         bias)
        cached = False
        if self.layer_type == "out":
            self.conv = APPNP(K=layer_index, alpha=0.1, cached=cached)
        else:
            self.conv = lambda x, y: x


class GPRGNNLayer(Layer):
    def __init__(self, in_channels=1, hid_channels=1, out_channels=1, layer_type="in", layer_index=2,
                 bias=True):
        super(GPRGNNLayer, self).__init__(in_channels, hid_channels, out_channels, layer_type, layer_index,
                                          bias)
        cached = False
        if self.layer_type == "out":
            self.conv = GPRProp(layer_index, alpha=0.1, cached=cached)
        else:
            self.conv = lambda x, y: x


class GCNIILayer(Layer):
    def __init__(self, in_channels, hid_channels, out_channels, layer_type, layer_index, bias=True, alpha=0.1,
                 theta=0.5):
        super(GCNIILayer, self).__init__(in_channels, hid_channels, out_channels, layer_type, layer_index)

        self.conv = GCN2Conv(channels=hid_channels, alpha=alpha, theta=theta, layer=layer_index + 1)


class GRANDLayer(Layer):
    def __init__(self, in_channels=1, hid_channels=1, out_channels=1, layer_type="in", layer_index=2,
                 bias=True):
        super(GRANDLayer, self).__init__(in_channels, hid_channels, out_channels, layer_type, layer_index,
                                         bias)
        cached = False
        self.conv = PlainConv(cached=cached)


class GPRProp(MessagePassing):
    '''
    propagation class for GPRGNN
    '''

    def __init__(self, K, alpha, cached=False, **kwargs):
        super(GPRProp, self).__init__(aggr='add', **kwargs)

        self.K = K
        self.alpha = alpha
        # PPR-like
        TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
        TEMP[-1] = (1 - alpha) ** K

        self.cached = cached
        self.edge_index = None
        self.norm = None

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
        self.temp.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, edge_index, edge_weight=None):
        if not self.cached or self.norm is None:
            self.edge_index, self.norm = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        edge_index = self.edge_index if self.edge_index is not None else edge_index
        norm = self.norm

        hidden = x * (self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class PlainConv(MessagePassing):
    def __init__(self, K=1, cached=True):
        super(PlainConv, self).__init__()
        self.edge_weight = None
        self.edge_index = None

        self.K = K
        self.cached = cached
        self.add_self_loops = True

        self._cached_x = None


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight = None) -> Tensor:
        
        cache = self._cached_x
        if cache is None:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)

            for k in range(self.K):
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                   size=None)
                if self.cached:
                    self._cached_x = x
        else:
            x = cache

        return x


    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)