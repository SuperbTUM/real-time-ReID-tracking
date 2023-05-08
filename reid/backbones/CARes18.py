import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from SERes18_IBN import GeM, IBN, trunc_normal_
from batchrenorm import BatchRenormalization2D
from attention_pooling import AttentionPooling


class CABlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CABlock, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)

        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        h, w = x.size(2), x.size(3)

        x_h = F.adaptive_avg_pool2d(x, (h, 1)).permute(0, 1, 3, 2)
        x_w = F.adaptive_avg_pool2d(x, (1, w))

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out


class CABasicBlock(nn.Module):
    def __init__(self, block, dim, renorm, ibn, se_attn, restride=False):
        super(CABasicBlock, self).__init__()
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
        self.cablock = CABlock(dim, se_attn)

    def forward(self, x):
        branch = x
        x = self.block_pre(x)
        x = self.cablock(x)
        if self.block_post:
            branch = self.block_post(branch)
        x += branch
        return F.relu(x)


class CARes18_IBN(nn.Module):
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

        self.basicBlock11 = CABasicBlock(model.layer1[0], 64, renorm, True, se_attn)

        self.basicBlock12 = CABasicBlock(model.layer1[1], 64, renorm, False, se_attn)

        self.basicBlock21 = CABasicBlock(model.layer2[0], 128, renorm, True, se_attn)

        self.basicBlock22 = CABasicBlock(model.layer2[1], 128, renorm, False, se_attn)

        self.basicBlock31 = CABasicBlock(model.layer3[0], 256, renorm, True, se_attn)

        self.basicBlock32 = CABasicBlock(model.layer3[1], 256, renorm, False, se_attn)

        # last stride = 1
        self.basicBlock41 = CABasicBlock(model.layer4[0], 512, renorm, True, se_attn, True)

        self.basicBlock42 = CABasicBlock(model.layer4[1], 512, renorm, False, se_attn)

        if pooling == "gem":
            self.avgpooling = GeM()
        else:
            self.avgpooling = model.avgpool

        # if renorm:
        #     self.bnneck = BatchRenormalization1D(512)
        #     self.bnneck.beta.requires_grad_(False)
        # else:
        #     self.bnneck = nn.BatchNorm1d(512)
        #     self.bnneck.bias.requires_grad_(False)

        self.bnneck = nn.BatchNorm1d(512)
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
        if self.is_reid:
            return feature
        x = self.bnneck(feature)
        if cam is not None:
            x = x + self.cam_factor * self.cam_bias[cam]
            trunc_normal_(x, std=0.02)
        x = self.classifier(x)

        return feature, x


def cares18_ibn(num_classes=751, pretrained="IMAGENET1K_V1", loss="triplet", **kwargs):
    if loss == "triplet":
        is_reid = False
    elif loss == "softmax":
        is_reid = True
    else:
        raise NotImplementedError
    model = CARes18_IBN(num_class=num_classes,
                        resnet18_pretrained=pretrained,
                        is_reid=is_reid,
                        **kwargs)
    return model
