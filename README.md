# graphia
Library of graph neural network implementations in PyTorch.

## Features

GNN Layers:
- Graph Convolutional Network (SpectralGraphConv) - [Kipf and Welling, 2017](https://arxiv.org/abs/1609.02907)
- GraphSAGE - [Hamilton and Ying et al., 2018](https://arxiv.org/abs/1706.02216)
- Graph Attention Network (GAT and MultiHeadGAT) - [Veličković et al., 2018](https://arxiv.org/abs/1710.10903)
- Residual Gated Graph ConvNets (GatedGraphConv) - [Bresson and Laurent, 2018](https://arxiv.org/abs/1711.07553)
- Graph Isomorphism Network (GIN) - [Xu and Hu et al., 2019](https://arxiv.org/abs/1810.00826)
- Convolutional ARMA Filters (ARMAConv) - [Bianchi et al., 2019](https://arxiv.org/abs/1901.01343)

Pooling:
- Global Sum/Ave/Max Pooling
- DiffPool - [Ying et al., 2019](https://arxiv.org/abs/1806.08804)
- MinCutPooling - [Bianchi and Grattarola et al., 2019](https://arxiv.org/abs/1907.00481)
- TopKPooling - [Cangea et al., 2018](https://arxiv.org/abs/1811.01287) and [Gao and Ji, 2019](https://arxiv.org/abs/1905.05178)


## Example Usage

To use graphia's GNN layers, simply import the layers you want to use and define a PyTorch model as you usually would. Below is an example of a graph neural network that maps an input graph with node dimensions N x 32 to a scalar real-valued output.

```python
import torch.nn as nn
from graphia.layers import GIN # Graph Isomorphism Network layer
from graphia.pooling import GlobalSumPooling


class GNNRegressor(nn.Module):

    def __init__(self):
        super(GNNRegressor, self).__init___()
        self.graph_layer_1 = GIN(in_features=32, out_features=32)
        self.graph_layer_2 = GIN(in_features=32, out_features=16)
        self.pooling = GlobalSumPooling()
        self.linear = nn.Linear(in_features=16, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, A, x):
        x = self.relu(self.graph_layer_1(A, x))
        x = self.relu(self.graph_layer_2(A, x))
        x = self.pooling(x)
        x = self.linear(x)
        return x

```