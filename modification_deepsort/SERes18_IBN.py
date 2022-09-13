import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F


class SEBlock(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.globalavgpooling = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(c_in, max(1, c_in // 16))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(max(1, c_in // 16), c_in)
        self.sigmoid = nn.Sigmoid()
        self.c_in = c_in

    def forward(self, x):
        assert self.c_in == x.size(1)
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
        Some do instance norm, some do batch norm
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


class SEDense18_IBN(nn.Module):
    """
    Additionally, we would like to test the network with local average pooling
    i.e. Divide into eight and concatenate them
    """
    def __init__(self, num_class=751, needs_norm=True, gem=True, is_reid=False, PAP=False):
        super().__init__()
        model = models.resnet18(pretrained=True)
        self.conv0 = model.conv1
        self.bn0 = model.bn1
        self.relu0 = model.relu
        self.pooling0 = model.maxpool

        model.layer1[0].bn1 = IBN(64)
        self.basicBlock11 = model.layer1[0]
        self.seblock1 = SEBlock(64)

        self.basicBlock12 = model.layer1[1]
        self.seblock2 = SEBlock(64)

        model.layer2[0].bn1 = IBN(128)
        self.basicBlock21 = model.layer2[0]
        self.seblock3 = SEBlock(128)
        self.ancillaryconv3 = nn.Conv2d(64, 128, 1, 2, 0)
        self.optionalNorm2dconv3 = nn.BatchNorm2d(128)

        self.basicBlock22 = model.layer2[1]
        self.seblock4 = SEBlock(128)

        model.layer3[0].bn1 = IBN(256)
        self.basicBlock31 = model.layer3[0]
        self.seblock5 = SEBlock(256)
        self.ancillaryconv5 = nn.Conv2d(128, 256, 1, 2, 0)
        self.optionalNorm2dconv5 = nn.BatchNorm2d(256)

        self.basicBlock32 = model.layer3[1]
        self.seblock6 = SEBlock(256)

        self.basicBlock41 = model.layer4[0]
        # last stride = 1
        self.basicBlock41.conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
                                            device="cuda:0")
        self.basicBlock41.downsample[0] = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False,
                                                    device="cuda:0")
        self.seblock7 = SEBlock(512)
        self.ancillaryconv7 = nn.Conv2d(256, 512, 1, 1, 0)
        self.optionalNorm2dconv7 = nn.BatchNorm2d(512)

        self.basicBlock42 = model.layer4[1]
        self.seblock8 = SEBlock(512)

        if gem:
            self.avgpooling = GeM()
        else:
            self.avgpooling = model.avgpool

        self.bnneck = nn.BatchNorm1d(512)
        self.bnneck.bias.requires_grad_(False)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_class),
        )
        self.needs_norm = needs_norm
        self.is_reid = is_reid
        self.PAP = PAP

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.pooling0(x)
        branch1 = x
        x = self.basicBlock11(x)
        scale1 = self.seblock1(x)
        x = scale1 * x + branch1

        branch2 = x
        x = self.basicBlock12(x)
        scale2 = self.seblock2(x)
        x = scale2 * x + branch2

        branch3 = x
        x = self.basicBlock21(x)
        scale3 = self.seblock3(x)
        if self.needs_norm:
            x = scale3 * x + self.optionalNorm2dconv3(self.ancillaryconv3(branch3))
        else:
            x = scale3 * x + self.ancillaryconv3(branch3)

        branch4 = x
        x = self.basicBlock22(x)
        scale4 = self.seblock4(x)
        x = scale4 * x + branch4

        branch5 = x
        x = self.basicBlock31(x)
        scale5 = self.seblock5(x)
        if self.needs_norm:
            x = scale5 * x + self.optionalNorm2dconv5(self.ancillaryconv5(branch5))
        else:
            x = scale5 * x + self.ancillaryconv5(branch5)

        branch6 = x
        x = self.basicBlock32(x)
        scale6 = self.seblock6(x)
        x = scale6 * x + branch6

        branch7 = x
        x = self.basicBlock41(x)
        scale7 = self.seblock7(x)
        if self.needs_norm:
            x = scale7 * x + self.optionalNorm2dconv7(self.ancillaryconv7(branch7))
        else:
            x = scale7 * x + self.ancillaryconv7(branch7)

        branch8 = x
        x = self.basicBlock42(x)
        scale8 = self.seblock8(x)
        x = scale8 * x + branch8

        if self.PAP:
            channels = x.size(-1)
            x_concatenate = []
            for i in range(8):
                x_concatenate.append(self.avgpooling(x[:, :, :, i*channels//8:min(channels, (i+1)*channels//8)].contiguous()))
            x = torch.cat(x_concatenate, dim=-1).cuda()
        else:
            x = self.avgpooling(x)
        feature = x.view(x.size(0), -1)
        if self.is_reid:
            return feature
        x = self.bnneck(feature)
        x = self.classifier(x)

        return x, feature
