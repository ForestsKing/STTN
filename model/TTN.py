import torch
from torch import nn

from model.atten import MultiHeadAttention


class TemporalTransformer(nn.Module):
    def __init__(self, d_k, d_v, d_model, len_his, n_heads, dropout):
        super(TemporalTransformer, self).__init__()
        self.D_T = nn.Parameter(torch.eye(len_his, len_his), requires_grad=True)
        self.conv = nn.Conv2d(len_his + d_model, d_model, (1, 1))

        self.atten = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # 时间位置嵌⼊层
        B, M, N, D = x.shape
        DT = self.D_T.unsqueeze(0).unsqueeze(2).repeat(B, 1, N, 1)
        x = torch.cat([x, DT], dim=-1)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 3, 2, 1)

        # Transform
        x = self.atten(x, x, x)
        x = x.permute(0, 2, 1, 3)

        residual = x.clone()
        x = self.feed_forward(x)
        x = self.dropout(self.norm(x + residual))
        return x
