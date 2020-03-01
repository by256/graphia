import torch
import torch.nn as nn


class SpectralGraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        """
        Kipf and Welling's graph convolution layer from (https://arxiv.org/abs/1609.02907).

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
            self.b = torch.nn.Parameter(torch.randn(self.out_features, ), requires_grad=True)
            nn.init.xavier_normal_(self.b.data)

    def forward(self, A, x):
        z = torch.matmul(A, torch.matmul(x, self.W)) 
        if self.bias:
            z = z + self.b
        return z


class GAT(nn.Module):
    """
    Petar Veličković's et al. Graph Attention layer from (https://arxiv.org/abs/1710.10903).

    Args:
        in_features (int): Number of features in each node of the input node feature matrix.
        out_features (int): Number of features in each node of the output node feature matrix.
        bias (boolean): Includes learnable additive bias if set to True
            Default: ``True``
    Attributes:
            W: learnable weight parameter of the transformation.
            b: learnable bias parameter of the transformation.
            attention_mechanism: single feedforward attention transformation layer.
            leaky_relu: LeakyReLU activation.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.W = torch.nn.Parameter(torch.randn(self.in_features, self.out_features), requires_grad=True)
        nn.init.xavier_normal_(self.W.data)
        self.attention_mechanism = nn.Linear(2*self.out_features, 1, bias=self.bias)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, A, x):
        Wh = torch.matmul(x, self.W)  # B x N x F_prime
        B, N = x.shape[0], x.shape[1]
        Wh_concat = torch.cat([Wh.view(B, N, 1, -1).repeat(1, 1, N, 1), Wh.view(B, 1, N, -1).repeat(1, N, 1, 1)], dim=-1)  # B x N x N x 2F_prime
        a = self.leaky_relu(self.attention_mechanism(Wh_concat)).squeeze()
        a = self.masked_softmax(a, A, dim=1)
        return torch.matmul(a, Wh)

    def masked_softmax(self, x, A, dim=3, epsilon=1e-5):
        exps = torch.exp(x)
        masked_exps = exps * A.float()
        masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
        return (masked_exps / masked_sums)


class MultiHeadGAT(nn.Module):
    """
    Petar Veličković's et al. Graph Attention layer from (https://arxiv.org/abs/1710.10903).

    Args:
        in_features (int): Number of features in each node of the input node feature matrix.
        head_out_features (int): Number of features in each node of the output node feature matrix in each head the multihead attention layer.
        n_heads (int): number of heads in the multihead attention layer.
            Default: 3
        multihead_agg (string): ``'concat'`` or ``'average'``. Aggregation function.
            Default: ``'concat'``
    Attributes:
            W: learnable weight parameter of the transformation.
            b: learnable bias parameter of the transformation.
            attention_mechanism: single feedforward attention transformation layer.
            leaky_relu: LeakyReLU activation.
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

