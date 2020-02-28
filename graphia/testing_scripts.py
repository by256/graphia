import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import UVvis
from pooling import GlobalMaxPooling, DiffPool
from layers import SpectralGraphConv



def test_uvvis_dataset():
    uvvis = UVvis(masked=True)
    x, node_masks, A, y = uvvis[0]
    print(x.shape, A.shape)
    gc = SpectralGraphConv(in_features=121, out_features=64)
    out = gc(A, x, node_masks)


class ToyModel(nn.Module):
    
    def __init__(self):
        super(ToyModel, self).__init__()
        self.gc1 = SpectralGraphConv(in_features=121, out_features=64)
        self.gc2 = SpectralGraphConv(in_features=64, out_features=64)
        self.gc3 = SpectralGraphConv(in_features=64, out_features=64)
        self.pooling = GlobalMaxPooling()
        self.linear = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, A, x, masks):
        x = self.relu(self.gc1(A, x))
        x = x * masks[:, :, :x.shape[-1]]
        x = self.relu(self.gc2(A, x))
        x = x * masks[:, :, :x.shape[-1]]
        x = self.relu(self.gc3(A, x))
        x = x * masks[:, :, :x.shape[-1]]

        x = self.pooling(x)
        x = self.linear(x)
        return x


def test_kipf_conv():
    train_uvvis = UVvis(masked=True)
    train_uvvis.df = train_uvvis.df.iloc[:1024, :]
    val_uvvis = UVvis(masked=True)
    val_uvvis.df = val_uvvis.df.iloc[1024:1280, :]
    train_loader = DataLoader(train_uvvis, batch_size=64)
    val_loader = DataLoader(val_uvvis, batch_size=64)

    model = ToyModel()
    optimizer = optim.Adam(model.parameters())
    MSE = nn.MSELoss()

    print(model, '\n')

    for epoch in range(256):
        train_losses = []
        train_metrics = []
        model.train()
        for idx, (A, x, masks, y_true) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(A, x, masks)
            loss = MSE(y_true, y_pred)
            train_losses.append(loss.item())
            train_metrics.append(nn.L1Loss()(y_true, y_pred).item())
            loss.backward()
            optimizer.step()

        val_losses = []
        val_metrics = []
        model.eval()
        for idx, (A, x, masks, y_true) in enumerate(val_loader):
            y_pred = model(A, x, masks)
            val_losses.append(MSE(y_true, y_pred).item())
            val_metrics.append(nn.L1Loss()(y_true, y_pred).item())
        
        print('Epoch: {}    Loss: {:.4f}    Train MAE: {:.4f}    Val Loss: {:.4f}    Val MAE: {:.4f}'.format(epoch+1, np.mean(train_losses), np.mean(train_metrics), np.mean(val_losses), np.mean(val_metrics)))


class ToyDiffPoolModel(nn.Module):
    
    def __init__(self):
        super(ToyDiffPoolModel, self).__init__()
        self.gc1 = SpectralGraphConv(in_features=121, out_features=64)
        self.diffpool_gc_embedding_1 = SpectralGraphConv(in_features=64, out_features=64, bias=False)
        self.diffpool_gc_pooling_1 = SpectralGraphConv(in_features=64, out_features=32, bias=False)
        self.diffpool_1 = DiffPool(self.diffpool_gc_embedding_1, self.diffpool_gc_pooling_1)

        self.gc2 = SpectralGraphConv(in_features=64, out_features=64)
        self.diffpool_gc_embedding_2 = SpectralGraphConv(in_features=64, out_features=64, bias=False)
        self.diffpool_gc_pooling_2 = SpectralGraphConv(in_features=64, out_features=16, bias=False)
        self.diffpool_2 = DiffPool(self.diffpool_gc_embedding_2, self.diffpool_gc_pooling_2)

        self.gc3 = SpectralGraphConv(in_features=64, out_features=64)
        self.diffpool_gc_embedding_3 = SpectralGraphConv(in_features=64, out_features=64, bias=False)
        self.diffpool_gc_pooling_3 = SpectralGraphConv(in_features=64, out_features=1, bias=False)
        self.diffpool_3 = DiffPool(self.diffpool_gc_embedding_3, self.diffpool_gc_pooling_3)

        self.linear = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, A, x, masks):
        x = self.relu(self.gc1(A, x))
        x = x * masks[:, :, :x.shape[-1]]
        A, x = self.diffpool_1(A, [A, x], [A, x])
        x = self.relu(x)

        x = self.relu(self.gc2(A, x))
        A, x = self.diffpool_2(A, [A, x], [A, x])
        x = self.relu(x)
        
        x = self.relu(self.gc3(A, x))
        A, x = self.diffpool_3(A, [A, x], [A, x])
        x = self.relu(x)
        
        x = self.linear(x).squeeze(-1)
        # print('out', x.shape)
        # print('\n\n')
        return x


def test_diffpool():
    train_uvvis = UVvis(masked=True)
    train_uvvis.df = train_uvvis.df.iloc[:1024, :]
    val_uvvis = UVvis(masked=True)
    val_uvvis.df = val_uvvis.df.iloc[1024:1280, :]
    train_loader = DataLoader(train_uvvis, batch_size=64)
    val_loader = DataLoader(val_uvvis, batch_size=64)

    model = ToyDiffPoolModel()
    optimizer = optim.Adam(model.parameters())
    MSE = nn.MSELoss()

    print(model, '\n')

    for epoch in range(256):
        train_losses = []
        train_metrics = []
        model.train()
        for idx, (A, x, masks, y_true) in enumerate(train_loader):
            # print('y_true', y_true.shape)
            optimizer.zero_grad()
            y_pred = model(A, x, masks)
            loss = MSE(y_true, y_pred)
            train_losses.append(loss.item())
            train_metrics.append(nn.L1Loss()(y_true, y_pred).item())
            loss.backward()
            optimizer.step()
            break
        break

        # val_losses = []
        # val_metrics = []
        # model.eval()
        # for idx, (A, x, masks, y_true) in enumerate(val_loader):
        #     y_pred = model(A, x, masks)
        #     val_losses.append(MSE(y_true, y_pred).item())
        #     val_metrics.append(nn.L1Loss()(y_true, y_pred).item())
        
        # print('Epoch: {}    Loss: {:.4f}    Train MAE: {:.4f}    Val Loss: {:.4f}    Val MAE: {:.4f}'.format(epoch+1, np.mean(train_losses), np.mean(train_metrics), np.mean(val_losses), np.mean(val_metrics)))



test_diffpool()