import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import Cora#, UVvis
from pooling import GlobalMaxPooling, GlobalSumPooling, DiffPool, MinCutPooling
from layers import SpectralGraphConv, GAT, MultiHeadGAT, GIN, ARMAConv



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
        self.diffpool_gc_embedding_1 = SpectralGraphConv(in_features=121, out_features=64, bias=False)
        self.diffpool_gc_pooling_1 = SpectralGraphConv(in_features=121, out_features=32, bias=False)
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

        self.pooling = GlobalMaxPooling()

    def forward(self, A, x, masks):
        # x = self.relu(self.gc1(A, x))
        # x = x * masks[:, :, :x.shape[-1]]
        # print(masks[0], '\n\n')
        # x = self.relu(self.gc2(A, x))
        # x = x * masks[:, :, :x.shape[-1]]
        # x = self.relu(self.gc3(A, x))
        # x = x * masks[:, :, :x.shape[-1]]
        # print('x', x.shape)
        A, x = self.diffpool_1(A, [A, x], [A, x])
        # x = x.squeeze()
        # print('x', x.shape, 'A', A.shape)
        # x = self.pooling(x)
        # print('x', x.shape)
        # x = self.relu(x)

        # x = self.relu(self.gc2(A, x))
        A, x = self.diffpool_2(A, [A, x], [A, x])
        # x = self.relu(x)
        
        # x = self.relu(self.gc3(A, x))
        A, x = self.diffpool_3(A, [A, x], [A, x])
        x = x.squeeze()
        # x = self.relu(x)
        
        x = self.linear(x)
        # print('out', x.shape)
        # print('\n\n')
        return x


def test_diffpool():
    train_uvvis = UVvis(masked=True)
    train_uvvis.df = train_uvvis.df.iloc[:4096, :]
    mu, sigma = np.mean(train_uvvis.df['computational']), np.std(train_uvvis.df['computational'])
    train_uvvis.df['computational'] = (train_uvvis.df['computational'] - mu) / sigma

    val_uvvis = UVvis(masked=True)
    val_uvvis.df = val_uvvis.df.iloc[4096:4096+256, :]
    val_uvvis.df['computational'] = (val_uvvis.df['computational'] - mu) / sigma

    train_loader = DataLoader(train_uvvis, batch_size=64)
    val_loader = DataLoader(val_uvvis, batch_size=64)

    model = ToyDiffPoolModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
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
            y_pred = (y_pred * sigma) + mu
            y_true = (y_true * sigma) + mu
            train_losses.append(loss.item())
            train_metrics.append(nn.L1Loss()(y_true, y_pred).item())
            loss.backward()
            optimizer.step()
        #     break
        # break

        val_losses = []
        val_metrics = []
        model.eval()
        for idx, (A, x, masks, y_true) in enumerate(val_loader):
            y_pred = model(A, x, masks)
            y_pred = (y_pred * sigma) + mu
            y_true = (y_true * sigma) + mu
            val_losses.append(MSE(y_true, y_pred).item())
            val_metrics.append(nn.L1Loss()(y_true, y_pred).item())
        
        print('Epoch: {}    Loss: {:.4f}    Train MAE: {:.4f}    Val Loss: {:.4f}    Val MAE: {:.4f}'.format(epoch+1, np.mean(train_losses), np.mean(train_metrics), np.mean(val_losses), np.mean(val_metrics)))



def get_uvvis_dataloaders(batch_size=4):
    train_uvvis = UVvis(masked=True)
    train_uvvis.df = train_uvvis.df.iloc[:4096, :]
    mu, sigma = np.mean(train_uvvis.df['computational']), np.std(train_uvvis.df['computational'])
    train_uvvis.df['computational'] = (train_uvvis.df['computational'] - mu) / sigma

    val_uvvis = UVvis(masked=True)
    val_uvvis.df = val_uvvis.df.iloc[4096:4096+256, :]
    val_uvvis.df['computational'] = (val_uvvis.df['computational'] - mu) / sigma

    train_loader = DataLoader(train_uvvis, batch_size=batch_size)
    val_loader = DataLoader(val_uvvis, batch_size=batch_size)
    return train_uvvis, val_uvvis, train_loader, val_loader


class ToyGATModel(nn.Module):
    
    def __init__(self):
        super(ToyGATModel, self).__init__()
        self.gat1 = MultiHeadGAT(in_features=121, head_out_features=64)
        self.gat2 = MultiHeadGAT(in_features=192, head_out_features=64)
        self.gat3 = MultiHeadGAT(in_features=192, head_out_features=64, multihead_agg='average')
        self.pooling = GlobalMaxPooling()
        self.linear = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, A, x, masks):
        x = self.relu(self.gat1(A, x))
        x = self.relu(self.gat2(A, x))
        x = self.relu(self.gat3(A, x))

        x = self.pooling(x)
        x = self.linear(x)
        return x


def test_GAT():

    n_train = 4096
    n_val = 256
    
    train_uvvis = UVvis(masked=True)
    train_uvvis.df = train_uvvis.df.iloc[:n_train, :]
    mu, sigma = np.mean(train_uvvis.df['computational']), np.std(train_uvvis.df['computational'])
    train_uvvis.df['computational'] = (train_uvvis.df['computational'] - mu) / sigma

    val_uvvis = UVvis(masked=True)
    val_uvvis.df = val_uvvis.df.iloc[n_train:n_train+n_val, :]
    val_uvvis.df['computational'] = (val_uvvis.df['computational'] - mu) / sigma

    train_loader = DataLoader(train_uvvis, batch_size=64)
    val_loader = DataLoader(val_uvvis, batch_size=64)
    
    model = ToyGATModel()

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
            y_pred = (y_pred * sigma) + mu
            y_true = (y_true * sigma) + mu
            train_losses.append(loss.item())
            train_metrics.append(nn.L1Loss()(y_true, y_pred).item())
            loss.backward()
            optimizer.step()
        #     break
        # break

        val_losses = []
        val_metrics = []
        model.eval()
        for idx, (A, x, masks, y_true) in enumerate(val_loader):
            y_pred = model(A, x, masks)
            y_pred = (y_pred * sigma) + mu
            y_true = (y_true * sigma) + mu
            val_losses.append(MSE(y_true, y_pred).item())
            val_metrics.append(nn.L1Loss()(y_true, y_pred).item())
        
        print('Epoch: {}    Loss: {:.4f}    Train MAE: {:.4f}    Val Loss: {:.4f}    Val MAE: {:.4f}'.format(epoch+1, np.mean(train_losses), np.mean(train_metrics), np.mean(val_losses), np.mean(val_metrics)))


class ToyGINModel(nn.Module):
    
    def __init__(self):
        super(ToyGINModel, self).__init__()
        self.gin1 = GIN(in_features=121, out_features=64)
        self.gin2 = GIN(in_features=64, out_features=64)
        self.gin3 = GIN(in_features=64, out_features=64)
        self.pooling = GlobalSumPooling()
        self.linear = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, A, x, masks):
        x = self.relu(self.gin1(A, x))
        x = x * masks[:, :, :x.shape[-1]]
        x = self.relu(self.gin2(A, x))
        x = x * masks[:, :, :x.shape[-1]]
        x = self.relu(self.gin3(A, x))
        x = x * masks[:, :, :x.shape[-1]]

        x = self.pooling(x)
        x = self.linear(x)
        return x


def test_GIN():

    n_train = 4096
    n_val = 256
    
    train_uvvis = UVvis(masked=True)
    train_uvvis.df = train_uvvis.df.iloc[:n_train, :]
    mu, sigma = np.mean(train_uvvis.df['computational']), np.std(train_uvvis.df['computational'])
    train_uvvis.df['computational'] = (train_uvvis.df['computational'] - mu) / sigma

    val_uvvis = UVvis(masked=True)
    val_uvvis.df = val_uvvis.df.iloc[n_train:n_train+n_val, :]
    val_uvvis.df['computational'] = (val_uvvis.df['computational'] - mu) / sigma

    train_loader = DataLoader(train_uvvis, batch_size=4)
    val_loader = DataLoader(val_uvvis, batch_size=4)

    model = ToyGINModel()
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
            y_pred = (y_pred * sigma) + mu
            y_true = (y_true * sigma) + mu
            train_losses.append(loss.item())
            train_metrics.append(nn.L1Loss()(y_true, y_pred).item())
            loss.backward()
            optimizer.step()
        #     break
        # break

        val_losses = []
        val_metrics = []
        model.eval()
        for idx, (A, x, masks, y_true) in enumerate(val_loader):
            y_pred = model(A, x, masks)
            y_pred = (y_pred * sigma) + mu
            y_true = (y_true * sigma) + mu
            val_losses.append(MSE(y_true, y_pred).item())
            val_metrics.append(nn.L1Loss()(y_true, y_pred).item())
        
        print('Epoch: {}    Loss: {:.4f}    Train MAE: {:.4f}    Val Loss: {:.4f}    Val MAE: {:.4f}'.format(epoch+1, np.mean(train_losses), np.mean(train_metrics), np.mean(val_losses), np.mean(val_metrics)))


class ToyGINCoraModel(nn.Module):
    
    def __init__(self):
        super(ToyGINCoraModel, self).__init__()
        # self.gc1 = GIN(in_features=1433, out_features=64)
        # self.gc2 = GIN(in_features=64, out_features=64)
        # self.gc3 = GIN(in_features=64, out_features=7)
        # self.gc1 = SpectralGraphConv(in_features=1433, out_features=64)
        # self.gc2 = SpectralGraphConv(in_features=64, out_features=64)
        # self.gc3 = SpectralGraphConv(in_features=64, out_features=7)
        # self.gc1 = GAT(in_features=1433, out_features=64)
        # self.gc2 = GAT(in_features=64, out_features=64)
        # self.gc3 = GAT(in_features=64, out_features=7)
        self.gc1 = ARMAConv(in_features=1433, out_features=64, timesteps=3)
        self.gc2 = ARMAConv(in_features=64, out_features=64, timesteps=3)
        self.gc3 = ARMAConv(in_features=64, out_features=7, timesteps=3)


        self.relu = nn.ReLU()

    def forward(self, A, x):
        x = self.relu(self.gc1(A, x))
        x = self.relu(self.gc2(A, x))
        x = F.log_softmax(self.gc3(A, x), dim=2)
        return x


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def test_cora():
    cora = Cora()
    A = cora.adj.to_dense().unsqueeze_(0)
    x = cora.features.unsqueeze_(0)
    labels = cora.labels#.unsqueeze_(0)
    idx_train = cora.idx_train
    idx_val = cora.idx_val

    print('A', A.shape)
    print('X', x.shape)
    print('labels', labels.shape)

    model = ToyGINCoraModel()
    optimizer = optim.Adam(model.parameters())
    nll = nn.NLLLoss()

    for epoch in range(400):
        train_losses = []
        train_metrics = []
        model.train()
        
        optimizer.zero_grad()
        y_pred = model(A, x).squeeze()
        loss = nll(y_pred[idx_train, :], labels[idx_train])
        train_losses.append(loss.item())
        train_metrics.append(accuracy(y_pred[idx_train], labels[idx_train]).item())
        loss.backward()
        optimizer.step()


        val_losses = []
        val_metrics = []
        model.eval()

        y_pred = model(A, x).squeeze()
        val_losses.append(nll(y_pred[idx_val, :], labels[idx_val]).item())
        val_metrics.append(accuracy(y_pred[idx_val], labels[idx_val]).item())
        
        print('Epoch: {}    Loss: {:.4f}    Train Accuracy: {:.4f}    Val Loss: {:.4f}    Val Accuracy: {:.4f}'.format(epoch+1, np.mean(train_losses), np.mean(train_metrics), np.mean(val_losses), np.mean(val_metrics)))


def test_mincut():

    cora = Cora(norm=False)
    A = cora.adj.to_dense().unsqueeze_(0)
    D = torch.diag(A.squeeze().sum(dim=-1))
    x = cora.features.unsqueeze_(0)
    print('A', A.shape)
    print('D', D.shape)
    print('x', x.shape)
    print(D)
    x = cora.features.unsqueeze_(0)

    mincut = MinCutPooling(1433, 64, 64)

    out = mincut(A, D, x)


test_mincut()