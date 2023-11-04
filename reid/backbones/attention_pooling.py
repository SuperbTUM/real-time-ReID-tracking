import torch
import torch.nn as nn
import torch.nn.functional as F
from .weight_init import trunc_normal_


class AttentionPooling(nn.Module):
    def __init__(self, height: int, width: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(height * width + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

        std = self.c_proj.in_features ** -0.5
        nn.init.normal_(self.q_proj.weight, std=std)
        nn.init.normal_(self.k_proj.weight, std=std)
        nn.init.normal_(self.v_proj.weight, std=std)
        nn.init.normal_(self.c_proj.weight, std=std)

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


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


class GeM3d(GeM):
    def __init__(self, kernel_size, stride=1, p=3, eps=1e-6):
        super(GeM3d, self).__init__(p, eps)
        self.kernel_size = kernel_size
        self.stride = stride

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool3d(x.clamp(min=eps).pow(p), self.kernel_size, self.stride).pow(1. / p)


class GeM_1D(GeM):
    def __init__(self, p=3, eps=1e-6):
        super(GeM_1D, self).__init__(p, eps)

    def gem(self, x, p=3, eps=1e-6):
        # return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)
        return F.adaptive_avg_pool1d(x.clamp(min=eps).pow(p), 1).pow(1. / p)
