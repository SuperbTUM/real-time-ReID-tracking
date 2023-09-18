from __future__ import division, absolute_import

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

import numpy as np
import math
import warnings
from timm.models.layers import trunc_normal_
from timm.models.layers import Mlp

from .attention_pooling import FeedForward, GeM_1D
from .weight_init import weights_init_classifier, weights_init_kaiming


__all__ = ["swin_t"]
pretrained_urls = {"swin_t": ""}


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


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


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


class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PostNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding, version="v1"):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted
        self.version = version

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if version == "v2":
            self.meta_mlp = Mlp(
                2,  # x, y
                hidden_features=384,
                out_features=heads,
                act_layer=nn.ReLU,
                drop=(0.125, 0.)  # FIXME should there be stochasticity, appears to 'overfit' without?
            )
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones(heads)))
            self._make_pair_wise_relative_positions()

        else:

            if self.relative_pos_embedding:
                self.relative_indices = get_relative_distances(window_size) + window_size - 1
                self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
            else:
                self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))
            trunc_normal_(self.pos_embedding, std=0.02)

        self.to_out = nn.Linear(inner_dim, dim)

        self.post_proj = nn.Linear(dim, dim)
        self.post_drop = nn.Dropout(0.1)

    def _make_pair_wise_relative_positions(self) -> None:
        """Method initializes the pair-wise relative positions to compute the positional biases."""
        device = self.logit_scale.device
        coordinates = torch.stack(torch.meshgrid([
            torch.arange(self.window_size, device=device),
            torch.arange(self.window_size, device=device)]), dim=0).flatten(1)
        relative_coordinates = coordinates[:, :, None] - coordinates[:, None, :]
        relative_coordinates = relative_coordinates.permute(1, 2, 0).reshape(-1, 2).float()
        relative_coordinates_log = torch.sign(relative_coordinates) * torch.log(
            1.0 + relative_coordinates.abs())
        self.register_buffer("relative_coordinates_log", relative_coordinates_log, persistent=False)

    def _relative_positional_encodings(self) -> torch.Tensor:
        """Method computes the relative positional encodings
        Returns:
            relative_position_bias (torch.Tensor): Relative positional encodings
            (1, number of heads, window size ** 2, window size ** 2)
        """
        window_area = self.window_size * self.window_size
        relative_position_bias = self.meta_mlp(self.relative_coordinates_log)
        relative_position_bias = relative_position_bias.transpose(1, 0).reshape(
            self.heads, window_area, window_area
        )
        relative_position_bias = relative_position_bias.unsqueeze(0)
        return relative_position_bias

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        if self.version == "v2":
            dots = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
            logit_scale = torch.clamp(self.logit_scale.reshape(1, self.heads, 1, 1, 1), max=math.log(1. / 0.01)).exp()
            dots = dots * logit_scale
            dots += self._relative_positional_encodings().unsqueeze(2)
        else:
            dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale
            if self.relative_pos_embedding:
                dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
            else:
                dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)
        out = self.post_proj(out)
        out = self.post_drop(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding, version="v1"):
        super().__init__()
        if version == "v2":
            self.attention_block = Residual(PostNorm(dim, WindowAttention(dim=dim,
                                                                          heads=heads,
                                                                          head_dim=head_dim,
                                                                          shifted=shifted,
                                                                          window_size=window_size,
                                                                          relative_pos_embedding=relative_pos_embedding,
                                                                          version=version)))
            self.mlp_block = Residual(PostNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))
        else:
            self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                         heads=heads,
                                                                         head_dim=head_dim,
                                                                         shifted=shifted,
                                                                         window_size=window_size,
                                                                         relative_pos_embedding=relative_pos_embedding,
                                                                         version=version)))
            self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class ShadowFeatureExtraction(nn.Module):
    def __init__(self, in_chan, hidden_dimension, camera=0, sequence=0, side_info=False, side_info_coeff=1.5):
        super(ShadowFeatureExtraction, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, 12, 2, stride=2)
        self.conv2 = nn.Conv2d(12, 48, 2, stride=2)
        self.norm = MixedNorm(12)
        self.fc = nn.Linear(48, hidden_dimension)
        if camera * sequence > 0:
            self.side_info_embedding = nn.Parameter(torch.randn(camera * sequence, 1, 1, hidden_dimension))
            trunc_normal_(self.side_info_embedding, std=0.02)
        elif camera > 0:
            self.side_info_embedding = nn.Parameter(torch.randn(camera, 1, 1, hidden_dimension))
            trunc_normal_(self.side_info_embedding, std=0.02)
        elif sequence > 0:
            self.side_info_embedding = nn.Parameter(torch.randn(sequence, 1, 1, hidden_dimension))
            trunc_normal_(self.side_info_embedding, std=0.02)
        self.side_info = side_info
        self.side_info_coeff = side_info_coeff

    def forward(self, x, view_index=None):
        x = F.relu(self.norm(self.conv1(x)))
        x = F.relu(self.conv2(x))
        flattened_x = x.permute(0, 2, 3, 1)  # BS, H, W, C
        flattened_output = self.fc(flattened_x)
        if self.side_info and view_index is not None:
            flattened_output += self.side_info_coeff * self.side_info_embedding[view_index]
        return flattened_output.permute(0, 3, 1, 2)


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding,
                 patch_merge=True,
                 version="v1"):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding, version=version),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding, version=version),
            ]))
        self.patch_merge = patch_merge

    def forward(self, x):
        if self.patch_merge:
            x = self.patch_partition(x)
        else:
            x = x.permute(0, 2, 3, 1)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)


class SwinTransformer(nn.Module):
    def __init__(self,
                 *,
                 hidden_dim,
                 layers,
                 heads,
                 loss="softmax",
                 channels=3,
                 num_classes=1000,
                 head_dim=32,
                 window_size=7,
                 downscaling_factors=(4, 2, 2, 2),
                 relative_pos_embedding=True,
                 camera=0,
                 sequence=0,
                 side_info=True,
                 version="v1",
                 gem=True):
        super().__init__()

        self.sfe = ShadowFeatureExtraction(channels, hidden_dim, camera=camera, sequence=sequence, side_info=side_info)

        self.stage1 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                                  patch_merge=False, version=version)
        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding, version=version)
        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding, version=version)
        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding, version=version)

        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        self.bottleneck = nn.BatchNorm1d(hidden_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, num_classes, bias=False)
        )
        self.mlp_head.apply(weights_init_classifier)

        self.img_channel_align = nn.Conv2d(hidden_dim, hidden_dim * 8, 8, stride=8)
        self.stage4_channel_align = nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, 4, 2, 1)
        self.stage3_channel_align = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1)
        self.stage2_channel_align = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1)

        if gem:
            self.avgpool = GeM_1D()
        else:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.loss = loss

    def forward(self, img, view_index=None):
        img = self.sfe(img, view_index)

        stage1_output = self.stage1(img)
        stage2_output = self.stage2(stage1_output)
        stage3_output = self.stage3(stage2_output)
        stage4_output = self.stage4(stage3_output)

        img_align = self.img_channel_align(img)
        fused_output = stage4_output + img_align
        stage3_align = self.stage4_channel_align(fused_output)
        fused_output = stage3_output + stage3_align
        stage2_align = self.stage3_channel_align(fused_output)
        fused_output = stage2_align + stage2_output
        stage1_align = self.stage2_channel_align(fused_output)
        fused_output = stage1_align + stage1_output

        fused_output = fused_output.view(fused_output.size(0), fused_output.size(1), -1).permute(0, 2, 1)  # (B, L, C)

        x = self.norm(fused_output)

        x = self.avgpool(x.permute(0, 2, 1)).squeeze()  # (B, C, 1)

        x_norm = self.bottleneck(x)
        y = self.mlp_head(x_norm)
        if not self.training:
            return y, x_norm
        if self.loss == "softmax":
            return y
        elif self.loss == "triplet":
            return y, x


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


def swin_t(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24),
           num_classes=751, pretrained=False, loss="softmax", **kwargs):
    model = SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, num_classes=num_classes, loss=loss, **kwargs)
    if pretrained:
        init_pretrained_weights(model, "swin_t")
    return model


if __name__ == "__main__":
    model = swin_t(loss="triplet").cuda()
    from torchsummary import summary
    try:
        print(summary(model, (3, 224, 224)))
    except:
        raise Warning("Model not ready yet!")
