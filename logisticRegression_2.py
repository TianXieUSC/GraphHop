import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy

from torch.nn import Linear
from torch.nn.parameter import Parameter
from run_model import TEMPERATURE, ALPHA, BETA, BATCH_PROP
from dataLoader import load_preprocess_data

learning_rate = 0.01
weight_decay = 1e-4
epoch = 1000
early_stopping = 20

temperature = TEMPERATURE
alpha = ALPHA
beta = BETA
batch_prop = BATCH_PROP


class LogisticRegression(torch.nn.Module):
    def __init__(self, num_feat, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = Linear(num_feat, num_classes)

    def forward(self, feat):
        return F.log_softmax(self.linear(feat), dim=1)

    def predict_soft_labels(self, feat):
        return F.softmax(self.linear(feat), dim=1)

    def predict_temp_soft_labels(self, feat):
        return F.softmax(self.linear(feat) / temperature, dim=1)

    def score(self, feat, labels):
        y_prob = F.softmax(self.linear(feat), dim=1)
        _accuracy = accuracy(y_prob, labels)
        return _accuracy


def fit(step, feat, labels, train_mask, val_mask, y_val, prev_model):
    num_feat = feat.shape[1]
    num_classes = labels.shape[1]
    pseudo_mask = torch.zeros(train_mask.shape[0], dtype=torch.bool)
    pseudo_mask[train_mask == False] = True

    X_train = feat[train_mask]
    y_train = labels[train_mask]
    X_pseudo = feat[pseudo_mask]
    y_pseudo = labels[pseudo_mask]

    X_val = feat[val_mask]
    y_val = torch.tensor(y_val)
    if step <= 1:
        model = LogisticRegression(num_feat, num_classes)
    else:
        model = prev_model
        # # add perturbations on model parameters
        # with torch.no_grad():
        #     for param in model.parameters():
        #         param.add_(torch.randn(param.size()) * 0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train()
    count = 0
    best_model = copy.deepcopy(model)
    prev_loss_val = np.inf
    for i in range(epoch):
        for j in range(0, int(1. / batch_prop) + 1):
            optimizer.zero_grad()
            y_train_batch = y_train[int(y_train.shape[0] * j * batch_prop):int(y_train.shape[0] * (j + 1) * batch_prop)]
            if y_train_batch.shape[0] == 0:
                break
            y_pseudo_batch = y_pseudo[int(y_pseudo.shape[0] * j * batch_prop):int(y_pseudo.shape[0] * (j + 1) * batch_prop)]
            y_train_log_prob = model.forward(X_train[int(X_train.shape[0] * j * batch_prop):int(X_train.shape[0] * (j + 1) * batch_prop)])
            y_pseudo_log_prob = model.forward(X_pseudo[int(X_pseudo.shape[0] * j * batch_prop):int(X_pseudo.shape[0] * (j + 1) * batch_prop)])
            num_train = y_train_batch.shape[0]
            num_pseudo = y_pseudo_batch.shape[0]
            if step == 0:
                entropy_train = (y_train_batch * y_train_log_prob).sum()
                loss_train = -1.0 * entropy_train
            else:
                entropy_train = (y_train_batch * y_train_log_prob).sum() / num_train + alpha * (
                        y_pseudo_batch * y_pseudo_log_prob).sum() / (num_pseudo * num_classes) \
                                + beta * (torch.exp(y_pseudo_log_prob) * y_pseudo_log_prob).sum() / (num_pseudo * num_classes)
                loss_train = -1.0 * entropy_train

            loss_train.backward()
            optimizer.step()

        if count == 0:
            best_model = copy.deepcopy(model)
        y_log_prob_val = model.forward(X_val)
        entropy_val = y_val * y_log_prob_val
        loss_val = -1.0 * entropy_val.sum()
        accuracy_val = accuracy(y_log_prob_val, y_val)
        if loss_val - prev_loss_val > 0 or prev_loss_val - loss_val < 1e-2:
            count += 1
        else:
            count = 0
        if count == early_stopping:
            break
        prev_loss_val = loss_val
        # print("epoch: {}, train loss: {:.4f}, train accuracy: {:.4f}, validation loss: {:.4f}, "
        #       "validation accuracy: {:.4f}".format(i, loss_train, accuracy_train, loss_val, accuracy_val))
        # f1.write(str(accuracy_val.item()) + ',\n')
    return best_model


def accuracy(output, labels):
    labels = labels.max(1)[1]
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# f1 = open('./results/output_amazon-photo.txt', 'w')
#
# dataset = 'cora'
# emb_dimensions = 20
# feat, one_hot_labels, adj, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_preprocess_data(dataset,
#                                                                                                           emb_dimensions)
#
# fit(0, torch.tensor(feat, dtype=torch.float), torch.tensor(one_hot_labels), train_mask, val_mask)

# # toy example
# from sklearn.datasets import load_iris
# from sklearn.preprocessing import OneHotEncoder
#
# X, y = load_iris(return_X_y=True)
# one_hot_y = OneHotEncoder().fit_transform(y.reshape(-1, 1))
# train_mask = np.ones(X.shape[0], dtype=bool)
#
# val_mask = train_mask
#
# temp = fit(torch.tensor(X, dtype=torch.float), torch.tensor(one_hot_y.toarray()), train_mask, val_mask)
# print(temp.score(torch.tensor(X, dtype=torch.float), torch.tensor(one_hot_y.toarray())))
