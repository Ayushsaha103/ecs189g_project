'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from code.base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp
import random

class Dataset_Loader(dataset):
    data = None
    dataset_name = None

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(Dataset_Loader, self).__init__(dName, dDescription)

    def adj_normalize(self, mx):
        """normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        # return torch.sparse.FloatTensor(indices, values, shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        onehot_labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return onehot_labels

    def normalize_features(self, features):
        """Row-normalize feature matrix"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return torch.FloatTensor(np.array(features.todense()))

    def load(self):
        """Load citation network dataset"""
        print('Loading {} dataset...'.format(self.dataset_name))

        # load node data from file
        idx_features_labels = np.genfromtxt("{}/node".format(self.dataset_source_folder_path), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        onehot_labels = self.encode_onehot(idx_features_labels[:, -1])

        # load link data from file and build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        reverse_idx_map = {i: j for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}/link".format(self.dataset_source_folder_path), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

        # Normalize features
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        features = self.normalize_features(features)  # Normalize features

        # convert to pytorch tensors
        #features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(onehot_labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        # the following part, you can either put them into the setting class or you can leave them in the dataset loader
        # the following train, test, val index are just examples, sample the train, test according to project requirements

        # generating lists of randomized indexes
        idx_train, idx_test = [], []
        idx_val = []            # (unused)

        labels_list = labels.tolist()
        uniq_labels_list = list(set(labels_list))
        train_cnt, test_cnt = -1, -1

        # set the num. of train/test elems (per class), for each kind of dataset
        if self.dataset_name == 'cora':
            train_cnt = 20
            test_cnt = 150
        elif self.dataset_name == 'citeseer':
            train_cnt = 20
            test_cnt = 200
        elif self.dataset_name == 'pubmed':
            train_cnt = 20
            test_cnt = 200
        # #---- cora-small is a toy dataset I handcrafted for debugging purposes ---
        # elif self.dataset_name == 'cora-small':
        #     idx_train = range(5)
        #     idx_val = range(5, 10)
        #     idx_test = range(5, 10)


        # randomize idx_train, idx_test
        for uniq in uniq_labels_list:
            class_indices = [index for index, value in enumerate(labels_list) if value == uniq]
            # add random indices to idx_train
            class_indices_sub = random.sample(class_indices, train_cnt)
            idx_train += class_indices_sub
            # add random indices to idx_test
            remaining_class_indices = list(set(class_indices) - set(class_indices_sub))
            idx_test += random.sample(remaining_class_indices, test_cnt)

        # convert idx_train, idx_test to torch.tensor's
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)        # (unused)
        idx_test = torch.LongTensor(idx_test)

        # get the training nodes/testing nodes
        # train_x = features[idx_train]
        # val_x = features[idx_val]
        # test_x = features[idx_test]
        # print(train_x, val_x, test_x)

        train_test_val = {'idx_train': idx_train, 'idx_test': idx_test, 'idx_val': idx_val}
        graph = {'node': idx_map, 'edge': edges, 'X': features, 'y': labels, 'utility': {'A': adj, 'reverse_idx': reverse_idx_map}}
        return {'graph': graph, 'train_test_val': train_test_val}
