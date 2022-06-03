from torch import nn


class Embedding(nn.Module):
    def __init__(self, d_feature, d_model):
        super(Embedding, self).__init__()
        self.conv = nn.Conv2d(d_feature, d_model, (1, 1))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x
