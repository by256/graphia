import numpy as np
import pandas as pd
import scipy.sparse as sp
# from rdkit import Chem
# from rdkit.Chem import rdmolops

import torch
from torch.utils.data import Dataset


class UVvis(Dataset):

    def __init__(self, masked=False):
        self.masked = masked
        self.df = pd.read_csv('../datasets/beard-uvvis.csv')
        self.clean_df()
        self.pad_dim = 55

    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles, comp, exp = row['smiles'], row['computational'], row['experimental']
        rd_mol = Chem.MolFromSmiles(smiles)
        node_features, node_mask = self.get_atomic_features(rd_mol)
        A = self.get_adjacency_matrix(rd_mol)

        output = torch.Tensor(A), torch.Tensor(node_features), torch.Tensor(node_mask), torch.Tensor([comp])#.squeeze()
        return output

    def get_adjacency_matrix(self, molecule):
        A = rdmolops.GetAdjacencyMatrix(molecule).astype(np.uint8)
        A = A + np.eye(A.shape[-1], dtype=np.uint8)
        h, w = A.shape[:2]
        if self.masked:
            padded_A = np.zeros(shape=(self.pad_dim, self.pad_dim))
            padded_A[:h, :w] = A
            A = padded_A
        # A = np.expand_dims(A, 0)
        return A

    def get_atomic_features(self, molecule):
        
        atoms = molecule.GetAtoms()

        if self.masked:
            n = self.pad_dim
        else:
            n = len(atoms)

        atomic_num_features = np.zeros(shape=(n, 100))
        charge_features = np.zeros(shape=(n, 3))
        ring_features = np.zeros(shape=(n, 1))
        aromatic_features = np.zeros(shape=(n, 1))
        valence_features = np.zeros(shape=(n, 6))
        degree_features = np.zeros(shape=(n, 4))
        hybridization_features = np.zeros(shape=(n, 4))
        chirality_features = np.zeros(shape=(n, 2))

        for i, atom in enumerate(atoms):
            Z = atom.GetAtomicNum()
            atomic_num_features[i, Z - 1] = 1
            # charge
            charge = atom.GetFormalCharge()
            if charge == -1:
                charge_features[i, 0] = 1
            elif charge == 0:
                charge_features[i, 1] = 1
            elif charge == 1:
                charge_features[i, 2] = 1
            # ring
            ring = int(atom.IsInRing())
            ring_features[i, 0] = ring
            # aromatic
            arom = atom.GetIsAromatic()
            aromatic_features[i, 0] = int(arom)
            # valence
            valence = atom.GetExplicitValence()
            valence_features[i, valence - 1] = 1
            # degree
            degree = atom.GetDegree()
            degree_features[i, degree - 1] = 1
            # hybridization
            hybridization = atom.GetHybridization()
            hybridization_features[i, hybridization - 1] = 1
            # chirality
            chirality = atom.GetChiralTag()
            if chirality > 0:
                chirality_features[i, chirality - 1] = 1

        atomic_features = np.concatenate([atomic_num_features, charge_features, ring_features, aromatic_features, valence_features, degree_features, hybridization_features, chirality_features], 1)
        
        pad_h, pad_w = len(atoms), atomic_features.shape[1]
        mask_vals = np.ones(shape=(pad_h, pad_w))
        mask = np.zeros_like(atomic_features)
        mask[:pad_h, :pad_w] = mask_vals

        return atomic_features, mask


    def clean_df(self):
        self.df = self.df.dropna(subset=['lambda1 (sTDA']).reset_index(drop=True)
        self.df = self.df.rename(columns={'SMI': 'smiles','lambda1 (sTDA': 'computational', ' nm).1': 'experimental'})
        self.df = self.df[['smiles', 'computational', 'experimental']]


class Cora(Dataset):
    """
    Adapted from (https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py)
    """
    def __init__(self, path='../datasets/cora/', laplacian=True):
        self.path = path
        self.laplacian = laplacian
        self.adj, self.features, self.labels, self.idx_train, self.idx_val, self.idx_test = self.build_graph()

    def build_graph(self):
        idx_features_labels = np.genfromtxt("{}cora.content".format(self.path), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = self.encode_onehot(idx_features_labels[:, -1])

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}cora.cites".format(self.path), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

            # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = self.normalize(features)
        adj = self.normalize(adj + sp.eye(adj.shape[0]))

        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, idx_train, idx_val, idx_test


    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return labels_onehot

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
