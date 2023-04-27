import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F

from .batchrenorm import BatchRenormalization2D, BatchRenormalization1D
from .attention_pooling import AttentionPooling


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# This can be applied as channel attention for gallery based on query
class SEBlock(nn.Module):
    def __init__(self, c_in, se_attn=False):
        super().__init__()
        if se_attn:
            self.globalavgpooling = GeM()
        else:
            self.globalavgpooling = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(c_in, max(1, c_in // 16))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(max(1, c_in // 16), c_in)
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
    def __init__(self, in_channels, ratio=0.5):
        """
        Half do instance norm, half do batch norm
        """
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.half = int(self.in_channels * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(self.in_channels - self.half)

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
            # block.conv1 = nn.Conv2d(dim >> 1, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            # block.downsample[0] = nn.Conv2d(dim >> 1, dim, kernel_size=(1, 1), stride=(1, 1), bias=False)
            block.conv1.stride = (1, 1)
            block.downsample[0].stride = (1, 1)
        if renorm:
            block.bn1 = BatchRenormalization2D(dim)
            block.bn2 = BatchRenormalization2D(dim)
        if ibn:
            # bn1 will be covered
            block.bn1 = IBN(dim)
        # block.relu = AconC(dim)
        if list(block.named_children())[-1][0] == "downsample":
            self.block_pre = nn.Sequential(*list(block.children())[:-1])
            self.block_post = block.downsample
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


class SERse18_IBN(nn.Module):
    """
    Additionally, we would like to test the network with local average pooling
    i.e. Divide into eight and concatenate them
    """
    def __init__(self,
                 resnet18_pretrained="IMAGENET1K_V1",
                 num_class=751,
                 num_cams=6,
                 needs_norm=True,
                 pooling="gem",
                 renorm=False,
                 se_attn=False,
                 is_reid=False):
        super().__init__()
        model = models.resnet18(weights=resnet18_pretrained, progress=False)
        self.conv0 = model.conv1
        if renorm:
            self.bn0 = BatchRenormalization2D(64)
        else:
            self.bn0 = model.bn1
        self.relu0 = model.relu
        self.pooling0 = model.maxpool

        self.basicBlock11 = SEBasicBlock(model.layer1[0], 64, renorm, True, se_attn)

        self.basicBlock12 = SEBasicBlock(model.layer1[1], 64, renorm, False, se_attn)

        self.basicBlock21 = SEBasicBlock(model.layer2[0], 128, renorm, True, se_attn)

        self.basicBlock22 = SEBasicBlock(model.layer2[1], 128, renorm, False, se_attn)

        self.basicBlock31 = SEBasicBlock(model.layer3[0], 256, renorm, True, se_attn)

        self.basicBlock32 = SEBasicBlock(model.layer3[1], 256, renorm, False, se_attn)

        # last stride = 1
        self.basicBlock41 = SEBasicBlock(model.layer4[0], 512, renorm, True, se_attn, True)

        self.basicBlock42 = SEBasicBlock(model.layer4[1], 512, renorm, False, se_attn)

        if pooling == "gem":
            self.avgpooling = GeM()
        else:
            self.avgpooling = model.avgpool

        self.bnneck = nn.BatchNorm1d(512)
        # self.bnneck = BatchRenormalization1D(512)
        self.bnneck.bias.requires_grad_(False)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_class),
        )
        # self.needs_norm = needs_norm
        self.is_reid = is_reid
        # self.cam_bias = nn.Parameter(torch.randn(num_cams, 512))

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
        # if cam is not None:
        #     feature = feature + self.cam_bias[cam]
        if self.is_reid:
            return feature
        x = self.bnneck(feature)
        x = self.classifier(x)

        return feature, x


def seres18_ibn(num_classes=751, pretrained="IMAGENET1K_V1", loss="triplet", **kwargs):
    if loss == "triplet":
        is_reid = False
    elif loss == "softmax":
        is_reid = True
    else:
        raise NotImplementedError
    model = SERse18_IBN(num_class=num_classes,
                        resnet18_pretrained=pretrained,
                        is_reid=is_reid,
                        **kwargs)
    return model
