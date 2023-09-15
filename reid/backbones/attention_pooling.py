import torch
import torch.nn as nn
import torch.nn.functional as F
from .weight_init import trunc_normal_


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')

    def forward(self, x):
        return self.net(x)


class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.cls_vec = nn.Parameter(torch.randn(in_dim), requires_grad=True)
        trunc_normal_(self.cls_vec, 0.02)
        self.fc = FeedForward(in_dim, 256)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        bs = x.size(0)
        weights = torch.matmul(x.view(-1, x.shape[1]), self.cls_vec)
        weights = self.softmax(weights.view(x.shape[0], -1))
        x = torch.bmm(x.view(x.shape[0], x.shape[1], -1), weights.unsqueeze(-1)).squeeze()
        x = x + self.cls_vec
        x = self.fc(x)
        x = x + self.cls_vec
        x = x.squeeze()
        if bs == 1:
            x = x.unsqueeze(0)
        return x


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)
        return F.adaptive_avg_pool2d(x.clamp(min=eps).pow(p), (1, 1)).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class GeM_Custom(GeM):
    def __init__(self, dim, p=3, eps=1e-6):
        super(GeM_Custom, self).__init__(p, eps)
        self.dim = dim

    def gem(self, x, p=3, eps=1e-6):
        return x.clamp(min=eps).pow(p).mean(self.dim, keepdim=True).pow(1. / p)
