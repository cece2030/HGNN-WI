import torch.nn as nn
import math
import torch
import torch.nn.functional as F

import arguments
from utils import *

args = arguments.parse_args()
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=True):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


#正负样本对比
class model(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(model, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.gcn1 = GCN(n_in, n_h, activation)
        self.lin = nn.Linear(n_h, n_h)

    def forward(self, seq1, seq2, seq3, adj1, adj2, sparse):





        h_1 = self.gcn(seq1, adj1, sparse)
        h_2 = self.gcn(seq2, adj2, sparse)
        h_3 = self.gcn(seq3, adj2, sparse)
        sc_1 = ((self.lin(h_1.squeeze(0))).sum(1))
        sc_2 = ((self.lin(h_2.squeeze(0))).sum(1))
        sc_3 = ((self.lin(h_3.squeeze(0))).sum(1))
        logits = torch.cat((sc_1, sc_2, sc_3), 0)


        return logits

    # Detach the return variables
    def embed(self, seq, adj, sparse):
        h_1 = self.gcn(seq, adj, sparse)
        h_2 = h_1.clone().squeeze(0)
        for i in range(1):
            h_2 = adj @ h_2

        h_2 = h_2.unsqueeze(0)

        return h_1.detach(), h_2.detach()

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=args.SemanticAttention_hidden):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

        self.beta = nn.Parameter(torch.rand(3, 1))

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)

        beta = torch.softmax(w+self.beta, dim=0)  # (M, 1)
        # beta = torch.sigmoid(w)

        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


# 定义一个注意力机制的类
class model_attention(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(model_attention, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.linear = nn.Linear(n_in, n_h)
        self.lin = nn.Linear(n_h, n_h)
        self.semantic_attention = SemanticAttention(in_size=n_in)

    def forward(self, semantic_embeddings, adj, sparse):
        # for i in range(len(semantic_embeddings)):
        #     semantic_embeddings[i] = self.linear(semantic_embeddings[i])

        x = self.semantic_attention(semantic_embeddings)
        # x = x.squeeze(0)
        adj_knn = get_knn_graph(x.squeeze(0), args.num_neighbor, knn_metric=args.knn_metric)
        adj_knn = process_adj(adj_knn)

        adj_knn = adj_knn.to(device)
        x_prop_aug = feature_propagation(adj_knn, x.squeeze(0), args.T, args.alpha)
        x = x.to(device)

        x_prop_aug = x_prop_aug.to(device)
        idx = np.random.permutation(x.shape[0])
        shuf_fts = x_prop_aug[idx, :]

        shuf_fts = shuf_fts.to(device)

        h_1 = self.gcn(x, adj, sparse)
        h_2 = self.gcn(shuf_fts, adj_knn, sparse)
        h_3 = self.gcn(x_prop_aug, adj_knn, sparse)
        sc_1 = ((self.lin(h_1.squeeze(0))).sum(1))
        sc_2 = ((self.lin(h_2.squeeze(0))).sum(1))
        sc_3 = ((self.lin(h_3.squeeze(0))).sum(1))
        logits = torch.cat((sc_1, sc_2, sc_3), 0)
        return x, x_prop_aug, adj_knn, adj, logits

    # Detach the return variables
    def embed(self, seq, adj, sparse):
        h_1 = self.gcn(seq, adj, sparse)
        h_2 = h_1.clone().squeeze(0)
        for i in range(1):
            h_2 = adj @ h_2

        h_2 = h_2.unsqueeze(0)

        return h_1.detach(), h_2.detach()