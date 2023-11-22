import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from collections import OrderedDict

from .weight_init import weights_init_kaiming, weights_init_classifier, trunc_normal_
from .batchrenorm import BatchRenormalization2D, BatchRenormalization1D, BatchRenormalization2D_Noniid
from .attention_pooling import GeM


# This can be applied as channel attention for gallery based on query
class SEBlock(nn.Module):
    def __init__(self, c_in: int, lbn: bool = False, renorm: bool = False):
        super().__init__()
        self.globalavgpooling = nn.AdaptiveAvgPool2d(1)
        mip = max(8, c_in // 16)
        self.fc1 = nn.Conv2d(c_in, mip, kernel_size=1, padding=0, bias=False)
        if lbn:
            self.bn = LBN_1D(mip, renorm=renorm)
        else:
            self.bn = nn.BatchNorm1d(mip)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(mip, c_in, bias=False)  # bias=False
        self.sigmoid = nn.Sigmoid()
        self.c_in = c_in
        self.lbn = lbn

        torch.nn.init.kaiming_normal_(self.fc1.weight.data, a=0, mode='fan_out')
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, a=0, mode='fan_out')

    def forward(self, x):
        x = self.globalavgpooling(x)
        x = self.fc1(x)
        x = x.squeeze()
        # x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.sigmoid(x)
        return x


class LBN_1D(nn.Module):
    def __init__(self, in_channels, ratio=0.5, renorm=False, dict_state=None):
        """
        Half do layer norm, half do batch norm
        """
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.half = int(self.in_channels * ratio)
        self.LN = nn.LayerNorm(self.half)
        if renorm:
            self.BN = BatchRenormalization1D(self.in_channels - self.half, dict_state) # experimental
        else:
            self.BN = nn.BatchNorm1d(self.in_channels - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.LN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


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
    def __init__(self, block, dim, renorm, ibn, se_ibn, restride=False):
        super(SEBasicBlock, self).__init__()
        if restride:
            block.conv1.stride = (1, 1)
            block.downsample[0].stride = (1, 1)
        if renorm:
            if not ibn:
                block.bn1 = BatchRenormalization2D(dim, block.bn1.state_dict())
            block.bn2 = BatchRenormalization2D(dim, block.bn2.state_dict())
        if ibn & renorm:
            block.bn1.BN = BatchRenormalization2D(dim >> 1, block.bn1.BN.state_dict())
        if list(block.named_children())[-1][0] == "downsample":
            self.block_pre = nn.Sequential(OrderedDict(list(block.named_children())[:-1]))
            downsample_layer = list(block.downsample.children())
            self.block_post = nn.Sequential(OrderedDict(
                [("conv", downsample_layer[0]),
                 ("bn", BatchRenormalization2D(dim, downsample_layer[1].state_dict()) if renorm else downsample_layer[1])]
            ))
        else:
            self.block_pre = block
            self.block_post = None
        self.seblock = SEBlock(dim, se_ibn, renorm)

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
    def __init__(self, block, dim, renorm, ibn, se_ibn, restride=False):
        super(SEPreActBasicBlock, self).__init__(block, dim, renorm, ibn, se_ibn, restride)
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


class MetaAconC(nn.Module):
    r""" ACON activation (activate or not).
    # MetaAconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is generated by a small network
    # according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """

    def __init__(self, width, r=16):
        super().__init__()
        self.fc1 = nn.Conv2d(width, max(r, width // r), kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(max(r, width // r))
        self.fc2 = nn.Conv2d(max(r, width // r), width, kernel_size=1, stride=1, bias=True)
        self.bn2 = nn.BatchNorm2d(width)

        self.p1 = nn.Parameter(torch.randn(1, width, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, width, 1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        beta = torch.sigmoid(
            self.bn2(self.fc2(self.bn1(self.fc1(x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True))))))
        return (self.p1 * x - self.p2 * x) * torch.sigmoid(beta * (self.p1 * x - self.p2 * x)) + self.p2 * x


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
                 se_lbn=True,
                 is_reid=False,
                 cam_factor=-1.):
        super().__init__()
        # model = models.resnet18(weights=resnet18_pretrained, progress=False)
        model = torch.hub.load("XingangPan/IBN-Net", "resnet18_ibn_a", pretrained=True)
        self.conv0 = model.conv1
        if renorm:
            self.bn0 = BatchRenormalization2D(64, model.bn1.state_dict())
        else:
            self.bn0 = model.bn1
        self.relu0 = model.relu
        self.pooling0 = model.maxpool

        self.basicBlock11 = SEBasicBlock(model.layer1[0], 64, renorm, True, se_lbn)

        self.basicBlock12 = SEBasicBlock(model.layer1[1], 64, renorm, True, se_lbn)

        self.basicBlock21 = SEBasicBlock(model.layer2[0], 128, renorm, True, se_lbn)

        self.basicBlock22 = SEBasicBlock(model.layer2[1], 128, renorm, True, se_lbn)

        self.basicBlock31 = SEBasicBlock(model.layer3[0], 256, renorm, True, se_lbn)

        self.basicBlock32 = SEBasicBlock(model.layer3[1], 256, renorm, True, se_lbn)

        # last stride = 1
        self.basicBlock41 = SEBasicBlock(model.layer4[0], 512, renorm, False, False, True)

        self.basicBlock42 = SEBasicBlock(model.layer4[1], 512, renorm, False, False)

        if pooling == "gem":
            self.avgpooling = GeM()
        else:
            self.avgpooling = model.avgpool

        # if renorm:
        #     self.bnneck = BatchRenormalization1D(512)
        #     self.bnneck.beta.requires_grad_(False)
        # else:
        self.bnneck = nn.BatchNorm1d(512)
        self.bnneck.bias.requires_grad_(False)

        self.bnneck.apply(weights_init_kaiming)

        self.classifier = nn.Sequential(
            nn.Linear(512, num_class, bias=False),
        )
        self.classifier.apply(weights_init_classifier)
        self.is_reid = is_reid
        self.cam_bias = nn.Parameter(torch.randn(num_cams, 512))
        trunc_normal_(self.cam_bias, std=0.02)
        self.cam_factor = cam_factor

    def forward(self, x, cam=None):
        x = self.conv0(x)
        x = self.bn0(x)
        # x = self.relu0(x)
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

        x_normed = self.bnneck(feature)
        if cam is not None:
            x_normed = x_normed + self.cam_factor * self.cam_bias[cam]
        x = self.classifier(x_normed)
        if self.is_reid:
            return x
        if not self.training:
            return x_normed, x
        return feature, x


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
