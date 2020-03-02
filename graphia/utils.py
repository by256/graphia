import torch


def degree_matrix(A):
    dummy_A = (A > 0.0).float()
    degree = torch.sum(dummy_A, dim=2)
    return torch.diag_embed(degree)
    
def normalize_A(A):
    D = degree_matrix(A)
    D_power = D**(-0.5)
    D_power[D_power > 1e12] = 0.0
    return torch.matmul(D_power, torch.matmul(A, D_power))