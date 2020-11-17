import numpy as np
import scipy.sparse as sp
import networkx as nx
import sys
import pickle as pkl
import random
import os


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'])


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def split_by_fixed_training_data(labels, num_labels_per_class=1):
    num = labels.shape[0]
    num_labels = np.unique(labels).shape[0]
    idx_train = []
    class_cnt = np.zeros(num_labels)
    for i in range(num):
        if labels[i] == num_labels:
            continue
        if (class_cnt >= num_labels_per_class).all():
            break
        if class_cnt[labels[i]] == num_labels_per_class:
            continue
        idx_train.append(i)
        class_cnt[labels[i]] += 1

    idx_val = random.sample(set(range(num)) - set(idx_train) - set(np.where(labels == num_labels)[0]), 500)  # random sample 500 for validation
    idx_test = list(set(range(num)) - set(idx_train) - set(idx_val))  # the rest as testing

    train_mask = np.zeros((num,), dtype=bool)
    train_mask[np.array(idx_train)] = 1

    val_mask = np.zeros((num,), dtype=bool)
    val_mask[np.array(idx_val)] = 1

    test_mask = np.zeros((num,), dtype=bool)
    test_mask[np.array(idx_test)] = 1
    return train_mask, val_mask, test_mask


def load_nell_data(path, num_labels_per_class=1, dataset_str='nell', validation_size=500, whether_val=True):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    label_rate = 0.01
    for i in range(len(names)):
        with open(path + "ind.{}.{}.{}".format(dataset_str, label_rate, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    adj = nx.to_scipy_sparse_matrix(nx.from_dict_of_lists(graph))
    # adj = sp.csr_matrix(adj)
    test_idx_reorder = parse_index_file(path + "ind.{}.{}.test.index".format(dataset_str, label_rate))
    test_idx_range = np.sort(test_idx_reorder)

    # Find relation nodes, add them as zero-vecs into the right position
    test_idx_range_full = range(allx.shape[0], len(graph))
    tx_extended = sp.lil_matrix((len(test_idx_range_full), tx.shape[1]))
    tx_extended[test_idx_range - allx.shape[0], :] = tx
    tx = tx_extended
    ty_extended = np.zeros((len(test_idx_range_full), ty.shape[1]))
    ty_extended[test_idx_range - allx.shape[0], :] = ty
    ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # reindex the classes, since there are only 105 classes exist
    labeled_classes = np.unique(np.where(labels == 1)[1])
    labeled_nodes = np.where(labels == 1)[0]
    unlabeled_nodes = np.setdiff1d(np.arange(labels.shape[0]), labeled_nodes)
    labels = labels[:, labeled_classes]

    all_labels = np.zeros(labels.shape[0], dtype=np.int)
    all_labels[unlabeled_nodes] = labels.shape[1] + 1
    all_labels[labeled_nodes] = np.where(labels != 0)[1]

    train_mask, val_mask, test_mask = split_by_fixed_training_data(all_labels, num_labels_per_class)
    test_mask[unlabeled_nodes] = False

    print('train: ', labels[train_mask].shape)
    print('val: ', labels[val_mask].shape)
    print('test: ', labels[test_mask].shape)
    print('NELL dataset loaded.')
    return features.toarray(), labels, adj, train_mask, val_mask, test_mask
