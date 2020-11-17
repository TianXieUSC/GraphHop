import argparse
import logging

import numpy as np
import scipy.sparse as sp
import torch
import time
from logisticRegression import fit

from dataLoader import load_preprocess_data, load_cora_full, load_amazon_dataset, load_coauthor_dataset, \
    load_planetoid_datasets, preprocess_features
from load_reddit_dataset import load_reddit_data
from utils import edge_weight, multiHops, normalize, one_shot_edge_weight, random_walk_normalize, pure_k_hops, \
    sparse_mx_to_torch_sparse_tensor, accuracy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, f1_score
from run_model import W1, W2, DATASET, ALPHA, BETA, TEMPERATURE, BATCH_PROP, NUM_PER_CLASS
from sklearn.decomposition import PCA
from datetime import datetime
from load_amazon2m_dataset import load_amazon2m_data

if torch.cuda.is_available():
    print("Using CUDA.")

dataset = DATASET

# logging.basicConfig(filename="./results/output_tuning_{}.txt".format(dataset),
#                     level=logging.DEBUG)
# logging.info(
#     "TEMPERATURE: {}, ALPHA: {}, BETA: {}, W1: {}, W2: {}, Batch prop: {}".format(TEMPERATURE, ALPHA, BETA, W1, W2,
#                                                                                   BATCH_PROP))


date = datetime.now()
logging.basicConfig(
    filename="./results/output_label_rate_{}_tuning_{}.txt".format(NUM_PER_CLASS,
                                                                   dataset),
    level=logging.DEBUG)
logging.info(
    "Date: {}, TEMPERATURE: {}, ALPHA: {}, BETA: {}, W1: {}, W2: {}".format(date.strftime("%Y-%b-%d (%H:%M:%S.%f)"),
                                                                            TEMPERATURE, ALPHA, BETA, W1, W2))

num_labels_per_class = NUM_PER_CLASS
if dataset in ['cora', 'citeseer', 'pubmed']:
    feat, one_hot_labels, adj, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_planetoid_datasets(
        dataset, num_labels_per_class)
    # feat, one_hot_labels, adj, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_preprocess_data(dataset)
elif dataset == 'cora_full':
    feat, one_hot_labels, adj, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_cora_full(
        num_labels_per_class=num_labels_per_class)

    # # pca feature dimension reduction
    # pca = PCA(n_components=1000)
    # feat = pca.fit_transform(feat)
    # print("PCA dimension reduction done!")

elif dataset in ['amazon-computers', 'amazon-photo']:
    name = dataset.split('-')[1]
    feat, one_hot_labels, adj, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_amazon_dataset(dataset,
                                                                                                             name,
                                                                                                             num_labels_per_class=num_labels_per_class)
elif dataset in ['coauthor-cs', 'coauthor-physics']:
    name = dataset.split('-')[1]
    feat, one_hot_labels, adj, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_coauthor_dataset(dataset,
                                                                                                               name,
                                                                                                               num_labels_per_class=num_labels_per_class)
elif dataset == 'reddit':
    print('Loading Reddit datasets...')
    path = "../../../datasets/reddit/"
    original_split = False
    feat, one_hot_labels, adj, train_mask, val_mask, test_mask = load_reddit_data(path, original_split,
                                                                                  num_labels_per_class=num_labels_per_class)

elif dataset == 'amazon2m':
    print('Loading Amazon2M datasets...')
    path = "./amazon2m"
    feat, one_hot_labels, adj, train_mask, val_mask, test_mask = load_amazon2m_data(path)

# feat = preprocess_features(feat)
labels = np.where(one_hot_labels == 1)[1]

print('train: ', feat[train_mask].shape)
print('val: ', feat[val_mask].shape)
print('test: ', feat[test_mask].shape)
print('edges: ', adj.count_nonzero())

# # change the size of training data
# train_index = np.random.choice(np.arange(feat.shape[0]), int(feat.shape[0] * 0.7), replace=False)
# test_index = np.setdiff1d(np.arange(feat.shape[0]), train_index)
# train_mask = np.zeros(feat.shape[0], dtype=bool)
# train_mask[train_index] = 1
# test_mask = np.zeros(feat.shape[0], dtype=bool)
# test_mask[test_index] = 1

print('data loaded.')
# k hops
one_hops_adj = pure_k_hops(adj, 1)
print('one hop matrix done')
# two_hops_adj = pure_k_hops(adj, 2)
# three_hops_adj = pure_k_hops(adj, 3)
# print('two hops matrix done.')
one_hops_adj = random_walk_normalize(one_hops_adj)
# two_hops_adj = random_walk_normalize(two_hops_adj)

# for one and two hops
new_feat = feat

pseudo_labels = np.zeros(one_hot_labels.shape)
pseudo_labels[train_mask] = one_hot_labels[train_mask]
y_val = one_hot_labels[val_mask]

epoch = 500
f1 = open('./results/output_{}_label_{}_temperature_{}_alpha_{}_w1_{}_w2_{}.txt'.format(dataset, num_labels_per_class,
                                                                                        TEMPERATURE, ALPHA, W1, W2),
          'a')
# f2 = open('./results/output_{}_two_hops_{}_weighted_{}_l2_norm.txt'.format(dataset, two_hops, weighted), 'w')

output = []
prev_model = [None, None]
num_perturb = 0
test_scores_record = []

new_feat = torch.FloatTensor(new_feat)
one_hops_adj = sparse_mx_to_torch_sparse_tensor(one_hops_adj)
# two_hops_adj = sparse_mx_to_torch_sparse_tensor(two_hops_adj)
y_val = torch.FloatTensor(y_val)
pseudo_labels = torch.FloatTensor(pseudo_labels)
one_hot_labels = torch.FloatTensor(one_hot_labels)

if torch.cuda.is_available():
    new_feat = new_feat.cuda()
    one_hops_adj = one_hops_adj.cuda()
    # two_hops_adj = two_hops_adj.cuda()
    pseudo_labels = pseudo_labels.cuda()
    one_hot_labels = one_hot_labels.cuda()

start_time = time.time()
for i in range(epoch + 1):
    one_agg_feat = torch.spmm(one_hops_adj, new_feat)
    # two_agg_feat = torch.spmm(two_hops_adj, new_feat)

    one_new_feat = torch.cat((new_feat, one_agg_feat), dim=1)
    # two_new_feat = torch.cat((new_feat, one_agg_feat, two_agg_feat), dim=1)

    X_1 = one_new_feat
    y_1 = pseudo_labels
    # X_2 = two_new_feat
    # y_2 = pseudo_labels

    if torch.cuda.is_available():
        X_1 = X_1.cuda()
        y_1 = y_1.cuda()
        # X_2 = X_2.cuda()
        # y_2 = y_2.cuda()
        y_val = y_val.cuda()

    clf_1 = fit(i, X_1, y_1, train_mask, val_mask, y_val, prev_model[0])
    # clf_2 = fit(i, X_2, y_2, train_mask, val_mask, y_val, prev_model[1])

    prev_model[0] = clf_1
    # prev_model[1] = clf_2
    if i != 0:
        pseudo_labels = W1 * pseudo_labels + (1. - W1) * clf_1.predict_temp_soft_labels(X_1).detach()
        new_feat = W2 * new_feat + (1. - W2) * clf_1.predict_soft_labels(X_1).detach()
    else:
        pseudo_labels = clf_1.predict_temp_soft_labels(X_1).detach()
        new_feat = clf_1.predict_soft_labels(X_1).detach()
    pseudo_labels[train_mask] = one_hot_labels[train_mask]

    # new_feat = W2 * new_feat + (1. - W2) * clf_1.predict_soft_labels(X_1).detach()

    # model evaluation
    clf_1.eval()

    y_train = one_hot_labels[train_mask]
    y_val = one_hot_labels[val_mask]
    y_test = one_hot_labels[test_mask]
    if torch.cuda.is_available():
        y_train = y_train.cuda()
        y_test = y_test.cuda()

    # train_score = (clf_1.score(X_1[train_mask], y_train) + clf_2.score(X_2[train_mask], y_train)) / 2
    # test_score = (clf_1.score(X_1[test_mask], y_test) + clf_2.score(X_2[test_mask], y_test)) / 2
    train_score = accuracy(new_feat[train_mask], y_train)
    val_score = accuracy(new_feat[val_mask], y_val)
    test_score = accuracy(new_feat[test_mask], y_test)
    print('epoch {}, train accuracy: {:.4f}, val accuracy: {:.4f}, test accuracy: {:.4f}'.format(i, train_score,
                                                                                                 val_score, test_score))
    logging.info('epoch: {}, train accuracy: {:.4f}, test accuracy: {:.4f}'.format(i, train_score, test_score))

    f1.write('{},'.format(i) + str(test_score.item()) + '\n')

    # output.append(new_feat[1708].tolist())
    # np.save('./results/output_{}_two_hops_{}_weighted_{}_test_1708_nodes.npy'.format(dataset, two_hops, weighted),
    #         np.array(output))

# logging.info('\n')
