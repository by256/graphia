# graphia
Library of graph neural network implementations in PyTorch.

## Features

GNN Layers:
- Graph Convolutional Network (GCN) - [Kipf and Welling](https://arxiv.org/abs/1609.02907)
- Graph Attention Network (GAT) - [Veličković et al.](https://arxiv.org/abs/1710.10903)
- Graph Isomorphism Network (GIN) - [Xu and Hu et al.](https://arxiv.org/abs/1810.00826)

Pooling:
- Global Sum/Max Pooling
- DiffPool - [Ying et al.](https://arxiv.org/abs/1806.08804)

## Example Usage

To use graphia's GNN layers, simply import the layers you want to use and define a PyTorch model as you usually would. Below is an example of a graph neural network that maps an input graph with node dimensions $N \times 32$ to a scalar real-valued output.

```python
import torch.nn
from graphia.layers import GIN # Graph Isomorphism Network layer
from graphia.pooling import GlobalSumPooling


class GraphNeuralNetwork(nn.Module):

    def __init__(self):
        super(GraphNeuralNetwork, self).__init___()
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