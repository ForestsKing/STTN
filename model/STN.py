import torch
from torch import nn

from model.atten import MultiHeadAttention
from model.gcn import GCN


class SpatialTransformer(nn.Module):
    def __init__(self, d_k, d_v, d_model, len_his, n_heads, n_nodes, dropout, adj=None):
        super(SpatialTransformer, self).__init__()
        self.adj = torch.Tensor(adj)
        if adj is None:
            self.D_S = nn.Parameter(torch.eye(n_nodes, n_nodes, requires_grad=True))
        else:
            self.D_S = nn.Parameter(self.adj, requires_grad=True)
        self.D_T = nn.Parameter(torch.eye(len_his, len_his), requires_grad=True)
        self.conv = nn.Conv2d(len_his + n_nodes + d_model, d_model, (1, 1))

        self.gcn = GCN(d_model, d_model, d_model * 4, dropout)

        self.atten = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )

        self.fs = nn.Linear(d_model, 1)
        self.fg = nn.Linear(d_model, 1)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # 空间-时间位置嵌⼊层
        B, M, N, D = x.shape
        DS = self.D_S.unsqueeze(0).unsqueeze(1).repeat(B, M, 1, 1)
        DT = self.D_T.unsqueeze(0).unsqueeze(2).repeat(B, 1, N, 1)
        x = torch.cat([x, DS, DT], dim=-1)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)

        # 静态图卷积层
        adj = self.adj.unsqueeze(0).unsqueeze(1).repeat(B, M, 1, 1)
        xg = self.gcn(x, adj)

        # 动态图卷积层
        y = self.atten(x, x, x)
        y_residual = y.clone()
        y = self.feed_forward(y)
        y = self.dropout(self.norm(y + y_residual))

        # ⻔机制
        g = torch.sigmoid(self.fs(y) + self.fg(xg))
        out = g * y + (1 - g) * xg

        return out
