import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralGraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        """
        (Kipf and Welling, 2017)'s graph convolution layer from (https://arxiv.org/abs/1609.02907).

        Args:
            in_features (int): Number of features in each node of the input node feature matrix.
            out_features (int): Number of features in each node of the output node feature matrix.
            bias (boolean): Includes learnable additive bias if set to True
                Default: ``True``

        Attributes:
            W: learnable weight parameter of the transformation.
            b: learnable bias parameter of the transformation.
        """
        super(SpectralGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.W = torch.nn.Parameter(torch.randn(self.in_features, self.out_features), requires_grad=True)
        nn.init.xavier_normal_(self.W.data)
        if self.bias:
            self.b = torch.nn.Parameter(torch.zeros(self.out_features, ), requires_grad=True)

    def forward(self, A, x):
        z = torch.matmul(A, torch.matmul(x, self.W)) 
        if self.bias:
            z = z + self.b
        return z


class GAT(nn.Module):
    """
    (Petar Veličković's et al., 2018)'s Graph Attention layer from (https://arxiv.org/abs/1710.10903).

    Args:
        in_features (int): Number of features in each node of the input node feature matrix.
        out_features (int): Number of features in each node of the output node feature matrix.
        bias (boolean): Includes learnable additive bias if set to True
            Default: ``True``
    Attributes:
            W: learnable weight parameter of the transformation.
            b: learnable bias parameter of the transformation.
            attention_mechanism: single feedforward attention transformation layer.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.W = torch.nn.Parameter(torch.randn(self.in_features, self.out_features), requires_grad=True)
        nn.init.xavier_normal_(self.W.data)
        self.attention_mechanism = nn.Linear(2*self.out_features, 1, bias=self.bias)

    def forward(self, A, x):
        Wh = torch.matmul(x, self.W)  # B x N x F_prime
        B, N = x.shape[0], x.shape[1]
        Wh_concat = torch.cat([Wh.view(B, N, 1, -1).repeat(1, 1, N, 1), Wh.view(B, 1, N, -1).repeat(1, N, 1, 1)], dim=-1)  # B x N x N x 2F_prime
        a = F.leaky_relu(self.attention_mechanism(Wh_concat), negative_slope=0.2).squeeze()
        a = self.masked_softmax(a, A, dim=1)
        return torch.matmul(a, Wh)

    def masked_softmax(self, x, A, dim=3, epsilon=1e-5):
        exps = torch.exp(x)
        masked_exps = exps * A.float()
        masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
        return (masked_exps / masked_sums)


class MultiHeadGAT(nn.Module):
    """
    (Petar Veličković's et al., 2018)'s Multi-Head Graph Attention layer from (https://arxiv.org/abs/1710.10903).

    Args:
        in_features (int): Number of features in each node of the input node feature matrix.
        head_out_features (int): Number of features in each node of the output node feature matrix in each head the multihead attention layer.
        n_heads (int): number of heads in the multihead attention layer.
            Default: 3
        multihead_agg (string): ``'concat'`` or ``'average'``. Aggregation function.
            Default: ``'concat'``
    """
    def __init__(self, in_features, head_out_features, n_heads=3, multihead_agg='concat'):
        super(MultiHeadGAT, self).__init__()
        self.in_features = in_features
        self.head_out_features = head_out_features
        self.n_heads = n_heads
        self.multihead_agg = multihead_agg

        for i in range(self.n_heads):
            setattr(self, 'GAT_head_{}'.format(i), GAT(self.in_features, self.head_out_features))
    
    def forward(self, A, x):
        if self.multihead_agg == 'concat':
            head_outputs = [getattr(self, 'GAT_head_{}'.format(i))(A, x) for i in range(self.n_heads)]
            h = torch.cat(head_outputs, dim=-1)
        elif self.multihead_agg == 'average':
            B, N = x.shape[0], x.shape[1]
            h = torch.zeros(size=(B, N, self.head_out_features))
            for i in range(self.n_heads):
                h += getattr(self, 'GAT_head_{}'.format(i))(A, x)
            h = h / self.n_heads
        else:
            raise ValueError('Multihead aggregation function must be either \'concat\' or \'average\'.')
        
        return h


class GIN(nn.Module):
    """
    (Xu and Hu et al., 2019)'s Graph Isomorphism Network from (https://arxiv.org/abs/1810.00826).
    
    Args:
        in_features (int): Number of features in each node of the input node feature matrix.
        out_features (int): Number of features in each node of the output node feature matrix.
    
    Attributes:
        epsilon (torch.nn.Parameter): learnable epsilon parameter.
        mlp (torch.nn.Linear): transformation function for aggregated node feature matrix.
    """
    def __init__(self, in_features, out_features):
        super(GIN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.epsilon = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        self.mlp = nn.Linear(self.in_features, self.out_features)

    def forward(self, A, x):
        return self.mlp((1 + self.epsilon)*x + torch.matmul(A, x))


class ARMAConvGCSLayer(nn.Module):
    """
    Graph Convolutional Skip (GCS) layer, which makes up a single element of the stack of GCS
    layers in an ARMAConv layer.

    Args:
        in_features (int): Number of features in each node of the input node feature matrix.
        out_features (int): Number of features in each node of the output node feature matrix.
        timesteps (int): number of recursive updates.
            Default: 1
        activation: torch.nn activation function used in each recursive update.
            Default: ``nn.ReLU()``

    Attributes:
        V_t (torch.nn.Parameter): Trainable parameter for input feature transformation.
        W_1 (torch.nn.Parameter): Trainable parameter for first iteration.
        W_t (torch.nn.Parameter): Trainable parameter for t-th iteration.
    """
    def __init__(self, in_features, out_features, timesteps=1, activation=nn.ReLU()):
        super(ARMAConvGCSLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.timesteps = timesteps
        self.activation = activation

        self.V_t = torch.nn.Parameter(torch.randn(self.in_features, self.out_features), requires_grad=True)
        nn.init.xavier_normal_(self.V_t.data)
        self.W_1 = torch.nn.Parameter(torch.randn(self.in_features, self.out_features), requires_grad=True)
        nn.init.xavier_normal_(self.W_1.data)
        if self.timesteps > 1:
            for i in range(2, self.timesteps+1):
                setattr(self, 'W_{}'.format(i), torch.nn.Parameter(torch.randn(self.out_features, self.out_features), requires_grad=True))


    def forward(self, L, x):
        x_t = x
        for i in range(1, self.timesteps+1):
            W_t = getattr(self, 'W_{}'.format(i))
            x_t = self.activation(torch.matmul(L, torch.matmul(x_t, W_t)) + torch.matmul(x, self.V_t))
        return x_t


class ARMAConv(nn.Module):
    """
    (Bianchi et al., 2019)'s Convolutional ARMA Filter from (https://arxiv.org/abs/1901.01343).

    Args:
        in_features (int): Number of features in each node of the input node feature matrix.
        out_features (int): Number of features in each node of the output node feature matrix.
        timesteps (int): Number of recursive updates.
            Default: 1
        k (int): Number of parallel stacks.
            Default: 3
        dropout_p (float): Dropout probability.
            Default: 0.2
    
    Attributes:
        GCS_k (ARMAConvGCSLayer): GCS layer of the k-th stack in the ARMAConv layer.
    """
    def __init__(self, in_features, out_features, timesteps=1, k=3, dropout_p=0.2):
        super(ARMAConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.timesteps = timesteps
        self.k = k
        self.dropout_p = dropout_p

        for i in range(1, self.k+1):
            setattr(self, 'GCS_{}'.format(i), ARMAConvGCSLayer(self.in_features, self.out_features, self.timesteps))

    def forward(self, L, x):
        B, N = x.shape[0], x.shape[1]
        X_out = torch.zeros(B, N, self.out_features)
        for i in range(1, self.k+1):
            gcs_layer = getattr(self, 'GCS_{}'.format(i))
            X_out += F.dropout(gcs_layer(L, x), p=0.2)
        return X_out / self.k


class GatedGraphConv(nn.Module):
    """
    (Bresson and Laurent, 2018)'s Gated Graph Convolution layer from (https://arxiv.org/abs/1711.07553).

    Args:
        in_features (int): Number of features in each node of the input node feature matrix.
        out_features (int): Number of features in each node of the output node feature matrix.

    Attributes:
        U (torch.nn.Parameter): Trainable parameter for input feature transformation.
        V (torch.nn.Parameter): Trainable parameter for input feature transformation.
        A (torch.nn.Parameter): Trainable parameter for edge gate transformation.
        B (torch.nn.Parameter): Trainable parameter for edge gate transformation.
    """
    def __init__(self, in_features, out_features):
        super(GatedGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Graph-conv params
        self.U = torch.nn.Parameter(torch.randn(self.in_features, self.out_features), requires_grad=True)
        nn.init.xavier_normal_(self.U.data)
        self.V = torch.nn.Parameter(torch.randn(self.in_features, self.out_features), requires_grad=True)
        nn.init.xavier_normal_(self.V.data)
        # Edge-gate params
        self.A = torch.nn.Parameter(torch.randn(self.in_features, self.out_features), requires_grad=True)
        nn.init.xavier_normal_(self.A.data)
        self.B = torch.nn.Parameter(torch.randn(self.in_features, self.out_features), requires_grad=True)
        nn.init.xavier_normal_(self.B.data)

    def forward(self, A, x):
        B, N = x.shape[0], x.shape[1]
        h_i = x.view(B, N, 1, -1).repeat(1, 1, N, 1)
        h_j = x.view(B, 1, N, -1).repeat(1, N, 1, 1)

        Ah_i = torch.matmul(h_i, self.A)
        Bh_j = torch.matmul(h_j, self.B)
        edge_gates = torch.sigmoid(Ah_i + Bh_j) * A.unsqueeze(-1)

        edge_gated_nbrs = torch.sum(edge_gates*torch.matmul(h_j, self.V), dim=2)
        return torch.matmul(x, self.U) + edge_gated_nbrs


class GraphSAGE(nn.Module):
    """
    (Hamilton and Ying et al., 2018)'s GraphSAGE layer from (https://arxiv.org/abs/1706.02216).

    Args:
        in_features (int): Number of features in each node of the input node feature matrix.
        out_features (int): Number of features in each node of the output node feature matrix.

    Attributes:
        linear (torch.nn.Linear): Fully connected dense transformation layer.
    """
    def __init__(self, in_features, out_features):
        super(GraphSAGE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(2*self.in_features, self.out_features)

    def forward(self, A, x):
        mean_aggregate = torch.matmul(A, x) / (torch.sum(A, dim=-1, keepdim=True) + 1e-6)
        h = torch.cat([x, mean_aggregate], dim=-1)
        return self.linear(h)