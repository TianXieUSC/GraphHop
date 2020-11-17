import networkx as nx
import json
import os
import numpy as np
import random
import copy

from networkx.readwrite import json_graph

input_path = ''
output_path = ''

def one_hot(idx, num_class):
    one_hot_labels = np.zeros((len(idx), num_class))
    one_hot_labels[np.arange(len(idx)), idx] = 1
    return one_hot_labels


def load_data(prefix, normalize=True):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n: int(n)
    else:
        conversion = lambda n: n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k): int(v) for k, v in id_map.items()}
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n: n
    else:
        lab_conversion = lambda n: int(n)

    class_map = {conversion(k): lab_conversion(v) for k, v in class_map.items()}

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    nodes = copy.deepcopy(G.nodes())
    for node in nodes:
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    edges = copy.deepcopy(G.edges())
    for edge in edges:
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
                G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    return G, feats, id_map, class_map


def split_by_fixed_training_data(labels, num_labels_per_class=20):
    num = labels.shape[0]
    num_labels = np.unique(labels).shape[0]

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

    train_mask = np.zeros((num,), dtype=bool)
    train_mask[np.array(idx_train)] = 1

    val_mask = np.zeros((num,), dtype=bool)
    val_mask[np.array(idx_val)] = 1

    test_mask = np.zeros((num,), dtype=bool)
    test_mask[np.array(idx_test)] = 1
    return train_mask, val_mask, test_mask


def load_reddit_data(path, num_labels_per_class=56):
    if os.path.exists(path + 'processed/'):
        path += 'processed/'
        G_data = json.load(open(path + 'G.json'))
        G = json_graph.node_link_graph(G_data)
        feats = np.load(path + 'feats.npy')
        id_map = json.load(open(path + 'id_map.json'))
        class_map = json.load(open(path + 'class_map.json'))
    else:
        G, feats, id_map, class_map = load_data(path + 'reddit')
        path += 'processed/'
        os.makedirs(path)

        G_data = json_graph.node_link_data(G)
        json.dump(G_data, open(path + 'G.json', 'w'))
        np.save(path + 'feats.npy', feats)
        json.dump(id_map, open(path + 'id_map.json', 'w'))
        json.dump(class_map, open(path + 'class_map.json', 'w'))

    # set the feature matrix and adjacency matrix the same order as in G.nodes()
    id_order = G.nodes()
    #     node_order = [id_map[i] for i in id_order]
    labels = np.array([class_map[i] for i in id_order])

    num_classes = np.unique(labels).shape[0]
    one_hot_labels = one_hot(labels, num_classes)

    # change the split of train / validation / test
    train_mask, val_mask, test_mask = split_by_fixed_training_data(labels, num_labels_per_class)
    for i, n in enumerate(G.nodes()):
        if train_mask[i]:
            G.node[n]['val'] = False
            G.node[n]['test'] = False
        elif val_mask[i]:
            G.node[n]['val'] = True
            G.node[n]['test'] = False
        elif test_mask[i]:
            G.node[n]['val'] = False
            G.node[n]['test'] = True

    # filter out the same nodes in adjacency matrix
    #     feats = feats[node_order]

    G_data = json_graph.node_link_data(G)
    json.dump(G_data, open(output_path + '-G.json', 'w'))
    np.save(output_path + '-feats.npy', feats)
    json.dump(id_map, open(output_path + '-id_map.json', 'w'))
    json.dump(class_map, open(output_path + '-class_map.json', 'w'))

    print('Reddit data loaded.')


if __name__ == '__main__':
    load_reddit_data(input_path)
