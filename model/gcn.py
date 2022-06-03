import torch
import torch.nn.functional as F
from torch import nn


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, last=False):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.last = last

        self.W = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)

    def forward(self, X, A):
        AX = torch.matmul(A, X)
        out = self.W(AX)

        if self.last:
            return out
        else:
            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            return out


class GCN(nn.Module):
    def __init__(self, C, F, H, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNLayer(C, H, dropout, False)
        self.conv2 = GCNLayer(H, F, dropout, True)

    def forward(self, X, A):
        hidden = self.conv1(X, A)
        out = self.conv2(hidden, A)
        return out
