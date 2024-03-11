import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def sample_indices(labels, num_samples_per_class, num_classes):
    np.random.seed(42)  # Seed for reproducibility
    idx_per_class = {i: [] for i in range(num_classes)}

    # Shuffle indices for each class
    for i in range(num_classes):
        idx = (labels == i).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(len(idx))]
        idx_per_class[i] = idx.tolist()

    # Sample indices for training and testing
    idx_train = []
    idx_test = []
    for i in range(num_classes):
        idx_train.extend(idx_per_class[i][:num_samples_per_class['train']])
        idx_test.extend(idx_per_class[i][-num_samples_per_class['test']:])

    return torch.LongTensor(idx_train), torch.LongTensor(idx_test)


def load_data(path, dataset="cora"):
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}/node".format(path),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}/link".format(path),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    num_classes = labels.max().item() + 1  # labels are 0-indexed

    # Define the number of samples per class for training and testing
    num_samples_per_class = {
        'train': 20,
        'test': 150
    }

    if dataset == 'citeseer':
        num_samples_per_class['test'] = 200
    elif dataset == 'pubmed':
        num_samples_per_class['test'] = 200

    # Randomly sample indices for the training and test sets
    idx_train, idx_test = sample_indices(labels, num_samples_per_class, num_classes)

    return adj, features, labels, idx_train, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sample_indices(labels, num_samples_per_class, num_classes):
    np.random.seed(42)  # Seed for reproducibility
    idx_per_class = {i: [] for i in range(num_classes)}

    # Shuffle indices for each class
    for i in range(num_classes):
        idx = (labels == i).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(len(idx))]
        idx_per_class[i] = idx.tolist()

    # Sample indices for training and testing
    idx_train = []
    idx_test = []
    for i in range(num_classes):
        idx_train.extend(idx_per_class[i][:num_samples_per_class['train']])
        idx_test.extend(idx_per_class[i][-num_samples_per_class['test']:])

    return torch.LongTensor(idx_train), torch.LongTensor(idx_test)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    #return torch.sparse.FloatTensor(indices, values, shape)
    return torch.sparse_coo_tensor(indices, values, shape)
