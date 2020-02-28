import torch
import torch.nn as nn


class GlobalMaxPooling(nn.Module):

    def __init__(self):
        super(GlobalMaxPooling, self).__init__()
    
    def forward(self, x):
        return torch.max(x, dim=1)[0]


class DiffPool(nn.Module):

    def __init__(self, embedding_gnn, pooling_gnn):
        super(DiffPool, self).__init__()
        self.embedding_gnn = embedding_gnn
        self.pooling_gnn = pooling_gnn
        self.softmax = nn.Softmax(dim=2)

    def forward(self, A, embedding_gnn_inputs, pooling_gnn_inputs):
        z = self.embedding_gnn(*embedding_gnn_inputs)
        # print('z', z.shape)
        s = self.softmax(self.pooling_gnn(*pooling_gnn_inputs))
        print('s', s.shape)
        print(s[0, 0, :])
        
        s_T = torch.transpose(s, dim0=1, dim1=2)
        # print('s_T', s_T.shape)

        A = torch.matmul(s_T, torch.matmul(A, s))
        
        x = torch.matmul(s_T, z)
        print('new A', A.shape)
        # print('new x', x.shape)
        print(A[0])
        print('\n\n')
        return A, x
