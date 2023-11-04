from __future__ import division, absolute_import

import math
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from timm.models.layers import trunc_normal_

from .weight_init import weights_init_kaiming, weights_init_classifier

import warnings
# helpers

__all__ = ["vit_t"]

pretrained_urls = {"vit_t": ""}


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

class MixedNorm(nn.Module):
    def __init__(self, features):
        super(MixedNorm, self).__init__()
        self.instancenorm = nn.InstanceNorm2d(features // 2, affine=True)
        self.batchnorm = nn.BatchNorm2d(features // 2)
        self.features = features

    def forward(self, x):
        split = torch.split(x, self.features // 2, 1)
        out1 = self.instancenorm(split[0].contiguous())
        out2 = self.batchnorm(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class Convolution_Stem(nn.Module):
    def __init__(self, in_chans=3, hidden_dim=64, embed_dim=384, stem_stride=1, patch_size=8):
        super(Convolution_Stem, self).__init__()
        self.stem_conv = nn.Sequential(
            nn.Conv2d(in_chans, hidden_dim, kernel_size=7, stride=stem_stride,
                      padding=3, bias=False),  # 112x112
            #  nn.BatchNorm2d(hidden_dim),
            MixedNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            #  nn.BatchNorm2d(hidden_dim),
            MixedNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(hidden_dim,
                              embed_dim,
                              kernel_size=patch_size // stem_stride,
                              stride=patch_size // stem_stride)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.proj(x) # B, C, H, W
        return x.flatten(2).permute(0, 2, 1) # B, H*W, C


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, loss="softmax", pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0., camera=0, sequence=0, side_info=True):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
        #     nn.LayerNorm(patch_dim),
        #     nn.Linear(patch_dim, dim),
        #     nn.LayerNorm(dim),
        # )
        self.to_patch_embedding = Convolution_Stem(in_chans=channels, stem_stride=2, embed_dim=dim, patch_size=patch_size)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        trunc_normal_(self.pos_embedding, std=0.02)
        if camera * sequence > 0:
            self.side_info_embedding = nn.Parameter(torch.randn(camera * sequence, 1, dim))
            trunc_normal_(self.side_info_embedding, std=0.02)
        elif camera > 0:
            self.side_info_embedding = nn.Parameter(torch.randn(camera, 1, dim))
            trunc_normal_(self.side_info_embedding, std=0.02)
        elif sequence > 0:
            self.side_info_embedding = nn.Parameter(torch.randn(sequence, 1, dim))
            trunc_normal_(self.side_info_embedding, std=0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        trunc_normal_(self.cls_token, std=0.02)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.LayerNorm(dim, eps=1e-6)

        self.bottleneck = nn.BatchNorm1d(dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.mlp_head = nn.Linear(dim, num_classes, bias=False)
        self.mlp_head.apply(weights_init_classifier)

        self.side_info = side_info
        self.loss = loss

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embedding", "cls_token"}

    def forward(self, img, view_index=None):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        if self.side_info and view_index is not None:
            x += 1.5 * self.side_info_embedding[view_index]
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.to_latent(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x_normed = self.bottleneck(x)

        y = self.mlp_head(x_normed)
        if not self.training:
            return y, x_normed
        if self.loss == "softmax":
            return y
        elif self.loss == "triplet":
            return y, x


def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


def init_pretrained_weights(model, key=''):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import os
    import errno
    import gdown
    from collections import OrderedDict

    def _get_torch_home():
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(
                    os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
                )
            )
        )
        return torch_home

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise
    filename = key + '_imagenet.pth'
    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        gdown.download(pretrained_urls[key], cached_file, quiet=False)

    state_dict = torch.load(cached_file)
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights from "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(cached_file)
        )
    else:
        print(
            'Successfully loaded imagenet pretrained weights from "{}"'.
            format(cached_file)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )


def vit_t(img_size=(224, 224), patch_size=32,
          num_classes=751, pretrained=False, loss="softmax", **kwargs):
    model = ViT(image_size=img_size, patch_size=patch_size, num_classes=num_classes, dim=384, depth=6, heads=16,
                mlp_dim=2048, dropout=0.1, emb_dropout=0.1, loss=loss, **kwargs)
    if pretrained:
        init_pretrained_weights(model, "vit_t")
    return model


if __name__ == "__main__":
    model = vit_t(loss="triplet").cuda()
    from torchsummary import summary
    try:
        print(summary(model, (3, 224, 224)))
    except:
        raise Warning("Model not ready yet!")
