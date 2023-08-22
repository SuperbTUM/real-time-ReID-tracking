import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
import math
from collections import OrderedDict

from .weight_init import weights_init_kaiming, weights_init_classifier
from .batchrenorm import BatchRenormalization2D, BatchRenormalization1D, BatchRenormalization2D_Noniid


# This can be applied as channel attention for gallery based on query
class SEBlock(nn.Module):
    def __init__(self, c_in, se_attn=False):
        super().__init__()
        if se_attn:
            self.globalavgpooling = GeM()
        else:
            self.globalavgpooling = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(c_in, max(1, c_in // 16), bias=False)  # bias=False
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(max(1, c_in // 16), c_in, bias=False)  # bias=False
        self.sigmoid = nn.Sigmoid()
        self.c_in = c_in

        torch.nn.init.kaiming_normal_(self.fc1.weight.data, a=0, mode='fan_out')
        torch.nn.init.constant_(self.fc1.bias.data, 0.0)
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, a=0, mode='fan_out')
        torch.nn.init.constant_(self.fc2.bias.data, 0.0)

    def forward(self, x):
        x = self.globalavgpooling(x)
        x = x.squeeze()
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.sigmoid(x)
        return x


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class IBN(nn.Module):
    def __init__(self, in_channels, ratio=0.5, renorm=False, non_iid=0, dict_state=None):
        """
        Half do instance norm, half do batch norm
        """
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.half = int(self.in_channels * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        if renorm:
            if non_iid:
                self.BN = BatchRenormalization2D_Noniid(self.in_channels - self.half, non_iid, dict_state)
            else:
                self.BN = BatchRenormalization2D(self.in_channels - self.half, dict_state) # experimental
        else:
            self.BN = nn.BatchNorm2d(self.in_channels - self.half)
        # experimental
        self.IN.weight.data.fill_(1.)
        self.IN.bias.data.zero_()

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class SEBasicBlock(nn.Module):
    def __init__(self, block, dim, renorm, ibn, se_attn, restride=False):
        super(SEBasicBlock, self).__init__()
        if restride:
            block.conv1.stride = (1, 1)
            block.downsample[0].stride = (1, 1)
        if renorm:
            if not ibn:
                block.bn1 = BatchRenormalization2D(dim, block.bn1.state_dict())
            block.bn2 = BatchRenormalization2D(dim, block.bn2.state_dict())
        if ibn:
            # bn1 will be covered
            # block.bn1 = IBN(dim)
            if renorm:
                block.bn1.BN = BatchRenormalization2D(dim >> 1, block.bn1.BN.state_dict())
        # block.relu = AconC(dim)
        if list(block.named_children())[-1][0] == "downsample":
            self.block_pre = nn.Sequential(OrderedDict(list(block.named_children())[:-1]))
            downsample_layer = list(block.downsample.children())
            self.block_post = nn.Sequential(OrderedDict(
                [("conv", downsample_layer[0]),
                 ("bn", downsample_layer[1])]
            ))
        else:
            self.block_pre = block
            self.block_post = None
        self.seblock = SEBlock(dim, se_attn)

    def forward(self, x):
        branch = x
        x = self.block_pre(x)
        scale = self.seblock(x)
        x = scale * x
        if self.block_post:
            branch = self.block_post(branch)
        x += branch
        return F.relu(x)


class SEPreActBasicBlock(SEBasicBlock):
    """I did not think of a proper pretrained weight to load
    """
    def __init__(self, block, dim, renorm, ibn, se_attn, restride=False):
        super(SEPreActBasicBlock, self).__init__(block, dim, renorm, ibn, se_attn, restride)
        in_planes = block.conv1.in_channels
        if renorm:
            block.bn1 = BatchRenormalization2D(in_planes)
        if ibn:
            block.bn1 = IBN(in_planes)
        self.block_pre = block

    def forward(self, x):
        branch = x
        x = self.block_pre.relu(self.block_pre.bn1(x))
        if self.block_post:
            # This is tricky in onnx inference as you need to disable conv fusion
            branch = self.block_post.conv(x)
        x = self.block_pre.conv1(x)
        x = self.block_pre.conv2(self.block_pre.relu(self.block_pre.bn2(x)))
        scale = self.seblock(x)
        x = scale * x
        x += branch
        return F.relu(x)


class AconC(nn.Module):
    r""" ACON activation (activate or not).
    # AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    # according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """

    def __init__(self, width):
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, width, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, width, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, width, 1, 1))

    def forward(self, x):
        return (self.p1 * x - self.p2 * x) * torch.sigmoid(self.beta * (self.p1 * x - self.p2 * x)) + self.p2 * x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class SERse18_IBN(nn.Module):
    """
    Additionally, we would like to test the network with local average pooling
    i.e. Divide into eight and concatenate them
    """
    def __init__(self,
                 num_class=751,
                 num_cams=6,
                 pooling="gem",
                 renorm=False,
                 se_attn=False,
                 is_reid=False):
        super().__init__()
        # model = models.resnet18(weights=resnet18_pretrained, progress=False)
        model = torch.hub.load("XingangPan/IBN-Net", "resnet18_ibn_a", pretrained=True)
        self.conv0 = model.conv1
        if renorm:
            self.bn0 = BatchRenormalization2D(64, self.bn0.state_dict())
        else:
            self.bn0 = model.bn1
        self.relu0 = model.relu
        self.pooling0 = model.maxpool

        self.basicBlock11 = SEBasicBlock(model.layer1[0], 64, renorm, True, se_attn)

        self.basicBlock12 = SEBasicBlock(model.layer1[1], 64, renorm, True, se_attn)

        self.basicBlock21 = SEBasicBlock(model.layer2[0], 128, renorm, True, se_attn)

        self.basicBlock22 = SEBasicBlock(model.layer2[1], 128, renorm, True, se_attn)

        self.basicBlock31 = SEBasicBlock(model.layer3[0], 256, renorm, True, se_attn)

        self.basicBlock32 = SEBasicBlock(model.layer3[1], 256, renorm, True, se_attn)

        # last stride = 1
        self.basicBlock41 = SEBasicBlock(model.layer4[0], 512, renorm, False, se_attn, True)

        self.basicBlock42 = SEBasicBlock(model.layer4[1], 512, renorm, False, se_attn)

        if pooling == "gem":
            self.avgpooling = GeM()
        else:
            self.avgpooling = model.avgpool

        if renorm:
            self.bnneck = BatchRenormalization1D(512)
            self.bnneck.beta.requires_grad_(False)
        else:
            self.bnneck = nn.BatchNorm1d(512)
            self.bnneck.bias.requires_grad_(False)

        self.bnneck.apply(weights_init_kaiming)

        self.classifier = nn.Sequential(
            nn.Linear(512, num_class, bias=False),
        )
        self.classifier.apply(weights_init_classifier)
        self.is_reid = is_reid
        self.cam_bias = nn.Parameter(torch.randn(num_cams, 512))
        self.cam_factor = 1.5

    def forward(self, x, cam=None):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.pooling0(x)

        x = self.basicBlock11(x)
        x = self.basicBlock12(x)
        x = self.basicBlock21(x)
        x = self.basicBlock22(x)
        x = self.basicBlock31(x)
        x = self.basicBlock32(x)
        x = self.basicBlock41(x)
        x = self.basicBlock42(x)

        x = self.avgpooling(x)
        feature = x.view(x.size(0), -1)
        if cam is not None:
            feature = feature + self.cam_factor * self.cam_bias[cam] # This is not good
            trunc_normal_(feature, std=0.02)
        x_normed = self.bnneck(feature)
        x = self.classifier(x_normed)
        if self.is_reid:
            return x
        if not self.training:
            return x_normed, x
        return feature, x_normed, x


def seres18_ibn(num_classes=751, loss="triplet", **kwargs):
    if loss == "triplet":
        is_reid = False
    elif loss == "softmax":
        is_reid = True
    else:
        raise NotImplementedError
    model = SERse18_IBN(num_class=num_classes,
                        is_reid=is_reid,
                        **kwargs)
    return model
