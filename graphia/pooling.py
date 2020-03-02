import torch
import torch.nn as nn
import torch.nn.functional as F


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
    (Ying et al., 2019)'s Hierarchical Graph Representation Learning with Differentiable Pooling (DiffPool) 
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


class MinCutPooling(nn.Module):
    """
    (Bianchi and Grattarola et al., 2019)'s minCUT pooling from (https://arxiv.org/abs/1907.00481).
    """
    def __init__(self, in_features, hidden_features, n_clusters):
        super(MinCutPooling, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.n_clusters = n_clusters

        self.W_1 = torch.nn.Parameter(torch.randn(self.in_features, self.hidden_features), requires_grad=True)
        nn.init.xavier_normal_(self.W_1.data)
        self.W_2 = torch.nn.Parameter(torch.randn(self.hidden_features, self.n_clusters), requires_grad=True)
        nn.init.xavier_normal_(self.W_2.data)



    def forward(self, A, D, x):
        S = F.softmax(torch.matmul(F.relu(torch.matmul(x, self.W_1)), self.W_2), dim=2).squeeze(1)
        print('S', S.shape)
        S_T = torch.transpose(S, dim0=1, dim1=2)

        
        A_pool = torch.matmul(S_T, torch.matmul(A, S))

        A_pool = A_pool - torch.eye(A_pool.shape[-1])*A_pool
        D_power = D**(-0.5)
        D_power[D_power > 1e12] = 0.0
        print(D_power)
        A_pool = torch.matmul(D_power, torch.matmul(A_pool, D_power))
        print(A_pool)

        # A_pool = torch.matmul(S, torch.matmul(A, S))
        X_pool = torch.matmul(S_T, x)
