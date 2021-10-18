import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial.distance import cosine, euclidean

from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

# select samples that are farthest to the center
from dataLoader import load_preprocess_data


def maxCover(data, ratio):
    center = np.mean(data, axis=0).reshape(1, -1)
    centers = np.tile(center, (center.shape[0], 1))
    distance = np.linalg.norm(data - centers, axis=1)
    sort_distance = np.argsort(distance)[::-1]
    return sort_distance[:int(len(sort_distance) * ratio)]


# select the top confidence
def topConfidence(data, ratio):
    row_max = np.max(data, axis=1)
    index = np.argsort(row_max)[::-1]
    return index[:int(len(index) * ratio)]


# assign each nodes to clusters
def nodesAssign(data, nodes_index, clusters):
    means = []
    for c in clusters:
        means.append(np.mean(data[c], axis=0))
    means = np.array(means)
    nodes = data[nodes_index]
    for i in range(len(nodes)):
        distance = np.linalg.norm(means - nodes[i], axis=1)
        min_c = np.argsort(distance)[0]
        clusters[min_c] = np.append(clusters[min_c], i)
    return clusters


# calculate the weight matrix
# w_ij = e^{d_ij / d_min} / sum_j(e^{d_ij / d_min})
# d_ij = cosine distance of i and j -- 1 - cos(a)
def edge_weight(data, adj, type='euclidean'):
    weight_matrix = np.zeros(adj.shape)
    diagonal = sp.diags(adj.diagonal())
    adj = adj - diagonal
    for i in range(data.shape[0]):
        adj_nodes_index = (adj[i].toarray() != 0).squeeze()
        adj_nodes = data[adj_nodes_index]

        # citeseer dataset has some isolated nodes
        if len(adj_nodes) == 0:
            continue
        assert (type in ['euclidean', 'cosine'])
        if type == 'euclidean':
            distance = np.array([euclidean(data[i], adj_nodes[j]) for j in range(adj_nodes.shape[0])])
        elif type == 'cosine':
            distance = np.array([cosine(data[i], adj_nodes[j]) for j in range(adj_nodes.shape[0])])

        # # there are some nodes have exactly same features.
        # if min(distance) == 0:
        #     index = np.where(distance == 0)
        #     index = np.arange(adj.shape[0])[adj_nodes_index][index]
        #     weight_matrix[i][index] = 1
        #     continue
        weight = np.exp(distance)
        weight = weight / np.sum(weight)
        weight_matrix[i][adj_nodes_index] = weight
    # weight_matrix += sp.eye(adj.shape[0])
    return weight_matrix


# calculate the feature similarity matrix
def one_shot_edge_weight(data, type='euclidean'):
    assert (type in ['euclidean', 'cosine'])
    if type == 'euclidean':
        distance = euclidean_distances(data)
    elif type == 'cosine':
        distance = cosine_similarity(data)
    return distance


# A = D^{-1/2} * A * D^{-1/2}
def normalize(adj):
    adj = adj + sp.eye(adj.shape[0])  # add self-loop
    row_sum = np.array(adj.sum(1))
    r_inv = np.power(row_sum, -0.5).flatten()
    r_mat_inv = sp.diags(r_inv)
    norm_adj = r_mat_inv.dot(adj)
    norm_adj = norm_adj.dot(r_mat_inv)
    return norm_adj


def random_walk_normalize(adj):
    # adj = adj + sp.eye(adj.shape[0])  # add self-loop
    row_sum = np.array(adj.sum(1)).astype('float')
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[r_inv == float('inf')] = 0
    r_mat_inv = sp.diags(r_inv)
    norm_adj = r_mat_inv.dot(adj)
    return norm_adj


# sparse adjacency matrix
def multiHops(adj, k):
    multi_adj = adj
    for i in range(k - 1):
        multi_adj = multi_adj.dot(adj)
    return multi_adj


def pure_k_hops(adj, k):
    multi_adj = adj
    pre_multi_adj = [adj]
    for i in range(k - 1):
        multi_adj = multi_adj.dot(adj)
        multi_adj = multi_adj - sp.diags(multi_adj.diagonal())
        multi_adj = multi_adj.tolil()
        for m in pre_multi_adj:
            multi_adj[m.nonzero()] = 0
        multi_adj = multi_adj.tocsr()
        pre_multi_adj.append(multi_adj)
    return multi_adj


# calculate the prediction accuracy
def accuracy(output, labels):
    labels = labels.max(1)[1]
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


if __name__ == '__main__':
    # # toy example
    # data = np.array([[1, 2, 3],
    #                  [4, 5, 6]])
    # adj = csr_matrix([[0, 1],
    #                   [1, 0]])
    # # adj = csr_matrix(np.array([[0, 1, 0, 0, 0, 0],
    # #                            [1, 0, 1, 0, 0, 0],
    # #                            [0, 1, 0, 1, 1, 0],
    # #                            [0, 0, 1, 0, 0, 0],
    # #                            [0, 0, 1, 0, 0, 1],
    # #                            [0, 0, 0, 0, 1, 0]]))
    # weight = np.array([[0., 1., 2., 1., 2, 3],
    #                    [1., 0., 1., 3., 4, 5],
    #                    [4., 1., 0., 1., 6, 1],
    #                    [1., 1., 1., 0., 3, 2],
    #                    [1, 2, 3, 4, 5, 0],
    #                    [4, 3, 2, 1, 6, 0]])
    #
    # a = edge_weight(data, adj)
    # print(a)

    dataset = 'pubmed'
    emb_dimensions = 20
    feat, one_hot_labels, adj, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_preprocess_data(dataset,
                                                                                                              emb_dimensions)
    a = edge_weight(feat, adj, 'cosine')
    print(a)
    # print(one_hot_labels.shape)
    # clf = LogisticRegression(solver="lbfgs", multi_class="multinomial", max_iter=1000, verbose=100)
    # labels = np.where(one_hot_labels == 1)[1]
    #
    # clf.fit(feat[train_mask], labels[train_mask])
    # print(clf.score(feat[train_mask], labels[train_mask]))

    # adj = adj + sp.eye(adj.shape[0])
    # weight = one_shot_edge_weight(feat, 'euclidean')
    # add_edge_weight(adj, weight)
