import torch
import torch.nn as nn


class GlobalMaxPooling(nn.Module):
    """
    Global max pooling layer. 
    Computes the node-wise maximum over the node feature matrix of a graph.
    """
    def __init__(self, dim=1):
        super(GlobalMaxPooling, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return torch.max(x, dim=self.dim)[0]


class GlobalSumPooling(nn.Module):
    """
    Global sum pooling layer. 
    Computes the node-wise summation over the node feature matrix of a graph.
    """
    def __init__(self, dim=1):
        super(GlobalSumPooling, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return torch.sum(x, dim=self.dim)


class GlobalAvePooling(nn.Module):
    """
    Global average pooling layer. 
    Computes the node-wise average over the node feature matrix of a graph.
    """
    def __init__(self, dim=1):
        super(GlobalAvePooling, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return torch.mean(x, dim=self.dim)


class MaskedAvePooling(nn.Module):
    """
    Masked average pooling layer. 
    Computes the node-wise average over the node feature matrix for nodes specified by a mask.
    """
    def __init__(self, dim=1):
        super(GlobalAvePooling, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        raise NotImplementedError


class DiffPool(nn.Module):
    """
    Ying et al. Hierarchical Graph Representation Learning with Differentiable Pooling (DiffPool) 
    pooling layer from (https://arxiv.org/abs/1806.08804).
    
    POTENTIALLY BROKEN CURRENTLY.

    Args:
        embedding_gnn: GNN layer for embedding layer within DiffPool.
        pooling_gnn: GNN layer for pooling layer within DiffPool.

    Attributes:
        softmax: torch.nn.Softmax activation function.
        relu: torch.nn.ReLU activation function

    """
    def __init__(self, embedding_gnn, pooling_gnn):
        super(DiffPool, self).__init__()
        self.embedding_gnn = embedding_gnn
        self.pooling_gnn = pooling_gnn
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()

    def forward(self, A, embedding_gnn_inputs, pooling_gnn_inputs):
        z = self.relu(self.embedding_gnn(*embedding_gnn_inputs))
        s = self.softmax(self.pooling_gnn(*pooling_gnn_inputs))
        s_T = torch.transpose(s, dim0=1, dim1=2)

        A = torch.matmul(s_T, torch.matmul(A, s))
        x = torch.matmul(s_T, z)
        return A, x
