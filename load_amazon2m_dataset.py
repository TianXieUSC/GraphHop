import networkx as nx
import json
import os
import numpy as np
import random
import copy

from networkx.readwrite import json_graph


def one_hot(idx, num_class):
    one_hot_labels = np.zeros((len(idx), num_class))
    one_hot_labels[np.arange(len(idx)), idx] = 1
    return one_hot_labels


def load_amazon2m_data(path):
    G_data = json.load(open(path + '-G.json'))
    G = json_graph.node_link_graph(G_data)
    feats = np.load(path + '-feats.npy')
    id_map = json.load(open(path + '-id_map.json'))
    class_map = json.load(open(path + '-class_map.json'))

    # set the feature matrix and adjacency matrix the same order as in G.nodes()
    id_order = G.nodes()
    node_order = [id_map[i] for i in id_order]
    labels = np.array([class_map[i] for i in id_order])

    num_classes = np.unique(labels).shape[0]
    one_hot_labels = one_hot(labels, num_classes)
    adj = nx.to_scipy_sparse_matrix(G)

    train_mask = np.zeros(feats.shape[0], dtype=bool)
    val_mask = np.zeros(feats.shape[0], dtype=bool)
    test_mask = np.zeros(feats.shape[0], dtype=bool)

    for node in G.nodes():
        if G.node[node]['val']:
            val_mask[id_map[node]] = True
        elif G.node[node]['test']:
            test_mask[id_map[node]] = True
        else:
            train_mask[id_map[node]] = True
    val_mask = val_mask[node_order]
    train_mask = train_mask[node_order]
    test_mask = test_mask[node_order]

    # filter out the same nodes in adjacency matrix
    feats = feats[node_order]
    print('Amazon2M data loaded.')
    print('train: ', feats[train_mask].shape)
    print('val: ', feats[val_mask].shape)
    print('test: ', feats[test_mask].shape)
    print('edges: ', adj.count_nonzero())
    return feats, one_hot_labels, adj, train_mask, val_mask, test_mask


if __name__ == '__main__':
    path = "/Users/tian/Documents/P8_Graph_Based_Learning/datasets/reddit/"
    feats, one_hot_labels, adj, train_mask, val_mask, test_mask = load_amazon2m_data(path, True)

    print('feat: ', feats.shape)
    print('labels: ', one_hot_labels.shape)
    print('adj: ', adj.shape)
    print('train mask: ', np.arange(feats.shape[0])[train_mask].shape)
    print('val mask: ', np.arange(feats.shape[0])[val_mask].shape)
    print('test mask: ', np.arange(feats.shape[0])[test_mask].shape)
