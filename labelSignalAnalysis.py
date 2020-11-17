import scipy.sparse as sp
import numpy as np
from dataLoader import load_planetoid_datasets
from utils import accuracy

dataset = 'pubmed'
num_labels_per_class = 20

feat, one_hot_labels, adj, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_planetoid_datasets(
    dataset, num_labels_per_class)

laplacian = sp.diags(np.squeeze(np.array(adj.sum(1)))) - adj

eigen_values, eigen_vectors = np.linalg.eig(laplacian.toarray())

# get the real part
eigen_values, eigen_vectors = np.real(eigen_values), np.real(eigen_vectors)

sort_index = np.argsort(eigen_values)
eigen_values = eigen_values[sort_index]
eigen_vectors = eigen_vectors[:, sort_index]


def label_signal_pred(eigen_vectors, reversed=True):
    if reversed:
        f1 = open('./results/label_signal_{}_{}.txt'.format(dataset, 'reversed'), 'a')
    else:
        f1 = open('./results/label_signal_{}.txt'.format(dataset), 'a')
    for i in range(1, 2000):
        top_k = i

        bases = eigen_vectors[:, :top_k]

        label_emb = np.dot(bases, np.dot(bases.T, one_hot_labels))

        preds = label_emb.argmax(1)
        labels = one_hot_labels.argmax(1)
        test_accuracy = np.mean(np.equal(preds, labels))
        print('index: {}, accuracy: {}'.format(i, test_accuracy))

        f1.write('{},'.format(i) + str(test_accuracy) + '\n')


label_signal_pred(eigen_vectors, False)
label_signal_pred(eigen_vectors[:, ::-1], True)
