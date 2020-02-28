import torch
import torch.nn as nn


class SpectralGraphConv(nn.Module):
    """Kipf's graph conv"""
    def __init__(self, in_features, out_features, bias=True):
        super(SpectralGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.W = torch.nn.Parameter(torch.randn(self.in_features, self.out_features), requires_grad=True)
        self.b = torch.nn.Parameter(torch.randn(self.out_features, ), requires_grad=True)

    def forward(self, A, x):
        z = torch.matmul(A, torch.matmul(x, self.W)) + self.b
        return z