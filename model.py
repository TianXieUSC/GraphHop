import numpy as np
import torch

from logisticRegression import fit
from datetime import datetime
from dataLoader import load_planetoid_datasets
from utils import random_walk_normalize, pure_k_hops, sparse_mx_to_torch_sparse_tensor, accuracy

from run_model import W1, W2, DATASET, NUM_PER_CLASS

if torch.cuda.is_available():
    print("Using CUDA.")

dataset = DATASET

date = datetime.now()

num_labels_per_class = NUM_PER_CLASS
if dataset in ['cora', 'citeseer', 'pubmed']:
    feat, one_hot_labels, adj, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_planetoid_datasets(
        dataset, num_labels_per_class)
else:
    assert False, "Choose dataset from Cora, CiteSeer, or PubMed."

labels = np.where(one_hot_labels == 1)[1]

# k hops
one_hops_adj = pure_k_hops(adj, 1)
two_hops_adj = pure_k_hops(adj, 2)

one_hops_adj = random_walk_normalize(one_hops_adj)
two_hops_adj = random_walk_normalize(two_hops_adj)

new_feat = feat

pseudo_labels = np.zeros(one_hot_labels.shape)
pseudo_labels[train_mask] = one_hot_labels[train_mask]
y_val = one_hot_labels[val_mask]

epoch = 100

output = []
prev_model = [None, None]
num_perturb = 0
test_scores_record = []

new_feat = torch.FloatTensor(new_feat)
one_hops_adj = sparse_mx_to_torch_sparse_tensor(one_hops_adj)
two_hops_adj = sparse_mx_to_torch_sparse_tensor(two_hops_adj)
y_val = torch.FloatTensor(y_val)
pseudo_labels = torch.FloatTensor(pseudo_labels)
one_hot_labels = torch.FloatTensor(one_hot_labels)

if torch.cuda.is_available():
    new_feat = new_feat.cuda()
    one_hops_adj = one_hops_adj.cuda()
    two_hops_adj = two_hops_adj.cuda()
    pseudo_labels = pseudo_labels.cuda()
    one_hot_labels = one_hot_labels.cuda()

ave_acc = []
for i in range(epoch + 1):
    one_agg_feat = torch.spmm(one_hops_adj, new_feat)
    two_agg_feat = torch.spmm(two_hops_adj, new_feat)

    one_new_feat = torch.cat((new_feat, one_agg_feat), dim=1)
    two_new_feat = torch.cat((new_feat, one_agg_feat, two_agg_feat), dim=1)

    X_1 = one_new_feat
    y_1 = pseudo_labels
    X_2 = two_new_feat
    y_2 = pseudo_labels

    if torch.cuda.is_available():
        X_1 = X_1.cuda()
        y_1 = y_1.cuda()
        X_2 = X_2.cuda()
        y_2 = y_2.cuda()
        y_val = y_val.cuda()

    clf_1 = fit(i, X_1, y_1, train_mask, val_mask, y_val, prev_model[0])
    clf_2 = fit(i, X_2, y_2, train_mask, val_mask, y_val, prev_model[1])

    prev_model[0] = clf_1
    prev_model[1] = clf_2

    pseudo_labels = W1 * clf_1.predict_temp_soft_labels(X_1).detach() + (1. - W1) * clf_2.predict_temp_soft_labels(
        X_2).detach()
    pseudo_labels[train_mask] = one_hot_labels[train_mask]

    new_feat = W2 * clf_1.predict_soft_labels(X_1).detach() + (1. - W2) * clf_2.predict_soft_labels(X_2).detach()

    # model evaluation
    clf_1.eval()
    clf_2.eval()

    y_train = one_hot_labels[train_mask]
    y_test = one_hot_labels[test_mask]
    if torch.cuda.is_available():
        y_train = y_train.cuda()
        y_test = y_test.cuda()

    train_score = accuracy(new_feat[train_mask], y_train)
    test_score = accuracy(new_feat[test_mask], y_test)
    print('Iteration {}, train accuracy: {:.4f}, test accuracy: {:.4f}'.format(i, train_score, test_score))
    ave_acc.append(test_score.item())
