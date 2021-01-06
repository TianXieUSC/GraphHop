import random

import numpy as np
import pickle as pkl
import sys
import scipy.sparse as sp
import networkx as nx
import os
import warnings

import torch
import torch_geometric.transforms as T

from torch_geometric.datasets import Planetoid

warnings.filterwarnings('ignore')

DATA_PATH = '../datasets'


def one_hot(idx, num_class):
    return torch.zeros(len(idx), num_class).to(idx.device).scatter_(
        1, idx.unsqueeze(1), 1.)


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(DATA_PATH + "/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file(DATA_PATH + "/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        # set zero if no label
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = np.vstack((np.array(allx.todense()), np.array(tx.todense())))
    labels = np.vstack((ally, ty))

    features[test_idx_reorder, :] = features[test_idx_range, :]
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    # from here mask the labels y for training, validation and testing,
    # so during training, only the labels from training dataset are used
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return features, labels, adj, y_train, y_val, y_test, train_mask, val_mask, test_mask


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    # print('Pre-processing feature by Simple Normalization')
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def preprocess_features_Probability(features):
    """Co-occurrence embedding to pre-process feature"""
    print('Pre-processing feature by Co-occurrence/Probability statistics')
    # co_occur = np.zeros((features.shape[1],features.shape[1]))

    # Get co-occurrence matrix
    co_occur = features.T.dot(features)

    # Normalization
    co_occur = preprocess_features(co_occur)
    features += features.dot(co_occur)
    features = preprocess_features(features)
    return features


def load_preprocess_data(dataset, emb_dimensions=20):
    features, labels, adj, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset)

    # drop the non-labeled nodes
    if dataset == 'citeseer':
        mask_index = []
        for i in range(len(labels)):
            if not (labels[i] == 0).all():
                mask_index.append(i)
        mask_index = np.array(mask_index)
        features = features[mask_index]
        labels = labels[mask_index]
        temp_adj = sp.csc_matrix(adj[mask_index]).T
        temp_adj = temp_adj[mask_index]
        adj = sp.csr_matrix(temp_adj)
        y_train = y_train[mask_index]
        y_val = y_val[mask_index]
        y_test = y_test[mask_index]
        train_mask = train_mask[mask_index]
        val_mask = val_mask[mask_index]
        test_mask = test_mask[mask_index]

    print("{} dataset loaded.".format(dataset))
    return features, labels, adj, y_train, y_val, y_test, train_mask, val_mask, test_mask


def split_by_fixed_training_data(data, num_labels_per_class=20):
    num = data.x.shape[0]
    num_labels = data.num_classes
    labels = data.y.tolist()
    # the different sampling of training set requires the parameters retuning
    # labels = np.random.RandomState(seed=2).permutation(labels).tolist()

    idx_train = []
    class_cnt = np.zeros(num_labels)
    for i in range(num):
        if (class_cnt >= num_labels_per_class).all():
            break
        if class_cnt[labels[i]] == num_labels_per_class:
            continue
        idx_train.append(i)
        class_cnt[labels[i]] += 1

    idx_val = random.sample(set(range(num)) - set(idx_train), 500)  # random sample 500 for validation
    idx_test = list(set(range(num)) - set(idx_train) - set(idx_val))  # the rest as testing

    train_mask = np.zeros((num,), dtype=int)
    train_mask[np.array(idx_train)] = 1

    val_mask = np.zeros((num,), dtype=int)
    val_mask[np.array(idx_val)] = 1

    test_mask = np.zeros((num,), dtype=int)
    test_mask[np.array(idx_test)] = 1
    return train_mask, val_mask, test_mask


def load_planetoid_datasets(dataset, num_labels_per_class=20):
    name = dataset
    path = os.path.join("./datasets", dataset)
    dataset = Planetoid(root=path, name=name, transform=T.NormalizeFeatures())
    data = dataset[0]
    data.num_classes = dataset.num_classes

    train_mask, val_mask, test_mask = split_by_fixed_training_data(data, num_labels_per_class)
    train_mask = train_mask.astype(bool)
    val_mask = val_mask.astype(bool)
    test_mask = test_mask.astype(bool)

    features = data.x.numpy()
    labels = one_hot(data.y, data.num_classes).numpy()
    edges = data.edge_index.numpy()
    ones = np.ones(edges.shape[1])
    adj = sp.csr_matrix((ones, edges), shape=(data.num_nodes, data.num_nodes))

    print("Citation-{} dataset loaded.".format(name))
    return features, labels, adj, None, None, None, train_mask, val_mask, test_mask


if __name__ == '__main__':
    features, labels, adj, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_planetoid_datasets('cora')
    print('train: ', features[train_mask].shape)
    print('val: ', features[val_mask].shape)
    print('test: ', features[test_mask].shape)
    print('labels: ', labels.shape)
    print(np.where(val_mask == True))
