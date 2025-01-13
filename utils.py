import copy
import os
import random
import arguments
import dgl
import psutil
import scipy.sparse as sp
import numpy as np
from scipy.sparse import csc_matrix, coo_matrix, dia_matrix
from sklearn.neighbors import kneighbors_graph
import torch
import torch.nn.functional as F
import torch_geometric
import torch as th
from torch import nn
from scipy.sparse import csr_matrix
from data_loader import data_loader
args = arguments.parse_args()
def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)
def sp_to_spt(mat):
    coo = mat.tocoo()  # 将输入稀疏矩阵mat转换为COO格式
    values = coo.data  # 从COO格式中获取矩阵的非零值
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))
def split(y, num_classes, train_per_class, val_per_class):

    indices = []

    for i in range(num_classes):
        index = (y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:train_per_class] for i in indices], dim=0)
    val_index = torch.cat([i[train_per_class:train_per_class+val_per_class] for i in indices], dim=0)
    test_index = torch.cat([i[train_per_class+val_per_class:] for i in indices], dim=0)

    train_mask = torch.zeros(y.size(), dtype=torch.bool)
    val_mask = torch.zeros(y.size(), dtype=torch.bool)
    test_mask = torch.zeros(y.size(), dtype=torch.bool)

    # 将对应的索引设为True
    train_mask[train_index] = True
    val_mask[val_index] = True
    test_mask[test_index] = True



    return train_index,val_index,test_index,train_mask, val_mask, test_mask
# def split_imdb(y, num_classes, train_per_class, val_per_class):
#     indices = [[] for _ in range(num_classes)]
#     for i, label in enumerate(y):
#         for label_idx, label_val in enumerate(label):
#             if label_val == 1:
#                 indices[label_idx].append(i)
#
#     train_indices, val_indices, test_indices = [], [], []
#     train_mask, val_mask, test_mask = torch.zeros(len(y)), torch.zeros(len(y)), torch.zeros(len(y))
#     for idxs in indices:
#         if len(idxs) < train_per_class + val_per_class:
#             train_indices.extend(idxs[:len(idxs) // 2])
#             val_indices.extend(idxs[len(idxs) // 2:len(idxs)])
#             test_indices.extend(idxs[len(idxs):])
#         else:
#             train_indices.extend(idxs[:train_per_class])
#             val_indices.extend(idxs[train_per_class:train_per_class + val_per_class])
#             test_indices.extend(idxs[train_per_class + val_per_class:])
#     train_mask = train_mask.type(torch.bool)
#     val_mask = val_mask.type(torch.bool)
#     test_mask = test_mask.type(torch.bool)
#     train_mask[train_indices] = True
#     val_mask[val_indices] = True
#     test_mask[test_indices] = True
#
#     return train_indices, val_indices, test_indices, train_mask, val_mask, test_mask

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch_geometric.seed_everything(seed)

def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def accuracy(output, labels):
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def feature_propagation(adj, features, K, alpha):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = features.to(device)
    adj = adj.to(device)
    features_prop = features.clone()
    for i in range(1, K + 1):

        features_prop = torch.sparse.mm(adj, features_prop)
        features_prop = (1 - alpha) * features_prop + alpha * features
    return features_prop.cpu()

def edge_index_to_sparse_mx(edge_index, num_nodes):
    edge_weight = np.array([1] * len(edge_index[0]))
    adj = csc_matrix((edge_weight, (edge_index[0], edge_index[1])),
                     shape=(num_nodes, num_nodes)).tolil()
    return adj

def process_adj(adj):
    adj.setdiag(1)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj



def get_knn_graph(x, num_neighbor, batch_size=0, knn_metric='cosine', connected_fast=True):
    if not batch_size:
        adj_knn = kneighbors_graph(x.cpu().detach().numpy(), num_neighbor, metric=knn_metric)

    else:
        if connected_fast:
            print('compute connected fast knn')
            num_neighbor1 = int(num_neighbor / 2)
            batches1 = get_random_batch(x.shape[0], batch_size)
            row1, col1 = global_knn(x, num_neighbor1, batches1, knn_metric)
            num_neighbor2 = num_neighbor - num_neighbor1
            batches2 = get_random_batch(x.shape[0], batch_size)
            row2, col2 = global_knn(x, num_neighbor2, batches2, knn_metric)
            row, col = np.concatenate((row1, row2)), np.concatenate((col1, col2))
        else:
            print('compute fast knn')
            batches = get_random_batch(x.shape[0], batch_size)
            row, col = global_knn(x, num_neighbor, batches, knn_metric)
        adj_knn = coo_matrix((np.ones_like(row), (row, col)), shape=(x.shape[0], x.shape[0]))
        print(adj_knn.shape)
        print(adj_knn)

    return adj_knn.tolil()

def get_random_batch(n, batch_size):
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    batches = []
    i = 0
    while i + batch_size * 2 < n:
        batches.append(idxs[i:i + batch_size])
        i += batch_size
    batches.append(idxs[i:])
    return batches

def global_knn(x, num_neighbor, batches, knn_metric):
    row = None
    for batch in batches:
        knn_current = kneighbors_graph(x[batch], num_neighbor, metric=knn_metric).tocoo()
        row_current = batch[knn_current.row]
        col_current = batch[knn_current.col]
        if row is None:
            row = row_current
            col = col_current
        else:
            row = np.concatenate((row, row_current))
            col = np.concatenate((col, col_current))
    return row, col



def get_feature_mask(rate, n_nodes, n_features):
    return torch.bernoulli(torch.Tensor([1 - rate]).repeat(n_nodes, n_features)).bool()
def load_acm(missing_feature):
    dl = data_loader('data/ACM')
    features = []

    for i in range(len(dl.nodes['count'])):
        th0 = dl.nodes['attr'][i]

        if th0 is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th0)

    link_type_dic = {0: 'pp', 1: '-pp', 2: 'pa', 3: 'ap', 4: 'ps', 5: 'sp', 6: 'pt', 7: 'tp'}
    paper_num = dl.nodes['count'][0]
    data_dic = {}

    for link_type in range(0, 8, 2):
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)
    print(hg)
    for etype in hg.canonical_etypes:
        edges_to_delete = np.random.choice(hg.num_edges(etype=etype), size=int(hg.num_edges(etype=etype) * args.missing_link),
                                           replace=False)
        edges_to_delete = torch.tensor(edges_to_delete, dtype=torch.int64)
        hg = dgl.remove_edges(hg, edges_to_delete, etype=etype)

    new_hg = dgl.heterograph({
        ('0', 'pa', '1'): (hg.edges(etype=hg.canonical_etypes[0])[0], hg.edges(etype=hg.canonical_etypes[0])[1]),
        ('0', 'pp', '0'): (hg.edges(etype=hg.canonical_etypes[1])[0], hg.edges(etype=hg.canonical_etypes[1])[1]),
        ('0', 'ps', '2'): (hg.edges(etype=hg.canonical_etypes[2])[0], hg.edges(etype=hg.canonical_etypes[2])[1]),
        ('0', 'pt', '3'): (hg.edges(etype=hg.canonical_etypes[3])[0], hg.edges(etype=hg.canonical_etypes[3])[1]),
        ('1', 'ap', '0'): (hg.edges(etype=hg.canonical_etypes[0])[1], hg.edges(etype=hg.canonical_etypes[0])[0]),
        ('0', '-pp', '0'): (hg.edges(etype=hg.canonical_etypes[1])[1], hg.edges(etype=hg.canonical_etypes[1])[0]),
        ('2', 'sp', '0'): (hg.edges(etype=hg.canonical_etypes[2])[1], hg.edges(etype=hg.canonical_etypes[2])[0]),
        ('3', 'tp', '0'): (hg.edges(etype=hg.canonical_etypes[3])[1], hg.edges(etype=hg.canonical_etypes[3])[0]),
    })
    print(new_hg)


    labels = dl.labels_test['data'][:paper_num] + dl.labels_train['data'][:paper_num]

    labels = [np.argmax(l) for l in labels]  # one-hot to value

    labels = th.LongTensor(labels)

    num_classes = dl.labels_train['num_classes']
    train_indices,valid_indices,test_indices,train_mask,valid_mask,test_mask = split(labels,num_classes,args.train_per_class,args.val_per_class)
    meta_paths = [['pa','ap'], ['ps', 'sp'], ['pp','pa','ap'],['pp', 'ps', 'sp'], ['-pp','pa','ap'],['-pp', 'ps', 'sp']]
    features[3] = features[3].toarray()
    for i in range(len(features)):
        if missing_feature > 0.0:

            feature_mask = get_feature_mask(rate=missing_feature, n_nodes=features[i].shape[0],
                                            n_features=features[i].shape[1])
            features[i] = torch.FloatTensor(features[i])
            features[i][~feature_mask] = 0.0
    features = [features.to(device)
                     for features in features]


    return new_hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
        th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths


def load_dblp(missing_feature):
    dl = data_loader('data/DBLP')
    features = []

    for i in range(len(dl.nodes['count'])):
        th0 = dl.nodes['attr'][i]

        if th0 is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th0)

    link_type_dic = {0: 'ap', 1: 'pt', 2: 'pv', 3: 'pa', 4: 'tp', 5: 'vp'}
    author_num = dl.nodes['count'][0]
    data_dic = {}

    for link_type in range(3):
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)
    print(hg)
    for etype in hg.canonical_etypes:
        edges_to_delete = np.random.choice(hg.num_edges(etype=etype), size=int(hg.num_edges(etype=etype) * args.missing_link),
                                           replace=False)
        edges_to_delete = torch.tensor(edges_to_delete, dtype=torch.int64)
        hg = dgl.remove_edges(hg, edges_to_delete, etype=etype)

    new_hg = dgl.heterograph({
        ('0', 'ap', '1'): (hg.edges(etype=hg.canonical_etypes[0])[0], hg.edges(etype=hg.canonical_etypes[0])[1]),
        ('1', 'pt', '2'): (hg.edges(etype=hg.canonical_etypes[1])[0], hg.edges(etype=hg.canonical_etypes[1])[1]),
        ('1', 'pv', '3'): (hg.edges(etype=hg.canonical_etypes[2])[0], hg.edges(etype=hg.canonical_etypes[2])[1]),
        ('1', 'pa', '0'): (hg.edges(etype=hg.canonical_etypes[0])[1], hg.edges(etype=hg.canonical_etypes[0])[0]),
        ('2', 'tp', '1'): (hg.edges(etype=hg.canonical_etypes[1])[1], hg.edges(etype=hg.canonical_etypes[1])[0]),
        ('3', 'vp', '1'): (hg.edges(etype=hg.canonical_etypes[2])[1], hg.edges(etype=hg.canonical_etypes[2])[0]),
    })

    print(new_hg)


    # author labels

    labels = dl.labels_test['data'][:author_num] + dl.labels_train['data'][:author_num]

    labels = [np.argmax(l) for l in labels]  # one-hot to value

    labels = th.LongTensor(labels)

    num_classes = dl.labels_train['num_classes']

    # train_valid_mask = dl.labels_train['mask'][:author_num]
    # test_mask = dl.labels_test['mask'][:author_num]
    # train_valid_indices = np.where(train_valid_mask == True)[0]
    # split_index = int(0.7 * np.shape(train_valid_indices)[0])
    # train_indices = train_valid_indices[:split_index]
    # valid_indices = train_valid_indices[split_index:]
    # train_mask = copy.copy(train_valid_mask)
    # valid_mask = copy.copy(train_valid_mask)
    # train_mask[valid_indices] = False
    # valid_mask[train_indices] = False
    # test_indices = np.where(test_mask == True)[0]

    meta_paths = [['ap', 'pa'], ['ap', 'pv', 'vp', 'pa'],['ap', 'pt', 'tp', 'pa']]
    train_indices, valid_indices, test_indices, train_mask, valid_mask, test_mask = split(labels, num_classes,
                                                                                          args.train_per_class,
                                                                                          args.val_per_class)

    features[3] = features[3].toarray()
    for i in range(len(features)):
        if missing_feature > 0.0:
            feature_mask = get_feature_mask(rate=missing_feature, n_nodes=features[i].shape[0],
                                            n_features=features[i].shape[1])
            features[i] = torch.FloatTensor(features[i])
            features[i][~feature_mask] = 0.0
        else:
            features[i] = torch.FloatTensor(features[i])
    features = [features.to(device)
                for features in features]
    return new_hg, features,labels, num_classes, train_indices, valid_indices, test_indices, \
        th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths


def load_imdb(missing_feature):
    dl = data_loader('data/IMDB')
    features = []

    for i in range(len(dl.nodes['count'])):
        th0 = dl.nodes['attr'][i]

        if th0 is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th0)

    link_type_dic = {0: 'md', 1: 'dm', 2: 'ma', 3: 'am', 4: 'mk', 5: 'km'}
    movie_num = dl.nodes['count'][0]
    data_dic = {}

    for link_type in range(0, 6, 2):
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])

        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)
    print(hg)
    for etype in hg.canonical_etypes:
        print(etype)
        edges_to_delete = np.random.choice(hg.num_edges(etype=etype), size=int(hg.num_edges(etype=etype) * args.missing_link),
                                           replace=False)
        edges_to_delete = torch.tensor(edges_to_delete, dtype=torch.int64)
        hg = dgl.remove_edges(hg, edges_to_delete, etype=etype)

    new_hg = dgl.heterograph({
        ('0', 'ma', '2'): (hg.edges(etype=hg.canonical_etypes[0])[0], hg.edges(etype=hg.canonical_etypes[0])[1]),
        ('0', 'md', '1'): (hg.edges(etype=hg.canonical_etypes[1])[0], hg.edges(etype=hg.canonical_etypes[1])[1]),
        ('0', 'mk', '3'): (hg.edges(etype=hg.canonical_etypes[2])[0], hg.edges(etype=hg.canonical_etypes[2])[1]),
        ('2', 'am', '0'): (hg.edges(etype=hg.canonical_etypes[0])[1], hg.edges(etype=hg.canonical_etypes[0])[0]),
        ('1', 'dm', '0'): (hg.edges(etype=hg.canonical_etypes[1])[1], hg.edges(etype=hg.canonical_etypes[1])[0]),
        ('3', 'km', '0'): (hg.edges(etype=hg.canonical_etypes[2])[1], hg.edges(etype=hg.canonical_etypes[2])[0]),
    })
    print(new_hg)



    # author labels










    labels0 = dl.labels_test['data'][:movie_num] + dl.labels_train['data'][:movie_num]

    labels = [np.argmax(l) for l in labels0]  # one-hot to value
    labels = th.LongTensor(labels)

    num_classes = dl.labels_train['num_classes']

    # train_valid_mask = dl.labels_train['mask'][:movie_num]
    # test_mask = dl.labels_test['mask'][:movie_num]
    # train_valid_indices = np.where(train_valid_mask == True)[0]
    # split_index = int(0.8 * np.shape(train_valid_indices)[0])
    # train_indices = train_valid_indices[:split_index]
    # valid_indices = train_valid_indices[split_index:]
    # train_mask = copy.copy(train_valid_mask)
    # valid_mask = copy.copy(train_valid_mask)
    # train_mask[valid_indices] = False
    # valid_mask[train_indices] = False
    # test_indices = np.where(test_mask == True)[0]
    train_indices, valid_indices, test_indices, train_mask, valid_mask, test_mask = split(labels, num_classes,
                                                                                          args.train_per_class,args.val_per_class)

    labels = labels0
    labels = th.LongTensor(labels)
    meta_paths = [['md', 'dm'], ['ma', 'am'], ['mk', 'km']]
    features[3] = features[3].toarray()
    for i in range(len(features)):
        if missing_feature > 0.0:
            feature_mask = get_feature_mask(rate=missing_feature, n_nodes=features[i].shape[0],
                                            n_features=features[i].shape[1])
            features[i] = torch.FloatTensor(features[i])
            features[i][~feature_mask] = 0.0
    features = [feature.to(device)
                for feature in features]
    return new_hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
        th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths

def show_info(): # 计算运行内存
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss / 1024 / 1024
    return memory



def union_adj_matrices(adj1, adj2):
    # 将稀疏张量转换为密集张量
    dense_adj1 = adj1.to_dense()
    dense_adj2 = adj2.to_dense()

    # 取两个邻接矩阵的和
    union_result = dense_adj1 + dense_adj2

    # 将大于1的值截断为1
    union_result[union_result > 1] = 1

    # 获取非零元素的索引和值
    indices = torch.nonzero(union_result)
    values = union_result[indices[:, 0], indices[:, 1]]

    # 将结果转换为稀疏张量
    union_result_sparse = torch.sparse.FloatTensor(indices.t(), values, union_result.size())

    return union_result_sparse


def process_adj_features(g, meta_paths, features, args):
    semantic_embeddings = []
    # 循环遍历元路径
    for mp in meta_paths:
        # 获取与元路径相关的子图
        sub_g = dgl.metapath_reachable_graph(g, mp)
        src, dst, _ = sub_g.cpu()._graph.edges(0)

        # 创建稀疏邻接矩阵
        adj = edge_index_to_sparse_mx(torch.stack([src, dst]), features[0].shape[0])
        adj = process_adj(adj)

        # 处理特征传播
        processed_features = feature_propagation(adj, features[0], args.T, args.alpha)

        semantic_embeddings.append(processed_features)

    return semantic_embeddings
def create_meta_adj_list(g, features, meta_paths):
    adj_list = []
    for mp in meta_paths:
        sub_g = dgl.metapath_reachable_graph(g, mp)
        src, dst, _ = sub_g.cpu()._graph.edges(0)
        adj = edge_index_to_sparse_mx(torch.stack([src, dst]), features[0].shape[0])
        adj = process_adj(adj)
        adj_list.append(adj)
    return adj_list

def union_adj_matrices(adj_list):
    # 将稀疏张量转换为密集张量
    adj = adj_list[0].to_dense()
    for i in range(1,len(adj_list)):

        adj = adj + adj_list[i].to_dense()


    # 将大于1的值截断为1
    adj[adj > 1] = 1

    # 获取非零元素的索引和值
    indices = torch.nonzero(adj)
    values = adj[indices[:, 0], indices[:, 1]]

    # 将结果转换为稀疏张量
    union_result_sparse = torch.sparse.FloatTensor(indices.t(), values, adj.size())

    return union_result_sparse
