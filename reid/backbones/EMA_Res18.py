import torch
from torch import nn
import torch.nn.functional as F

from .weight_init import weights_init_classifier, weights_init_kaiming, trunc_normal_
from .SERes18_IBN import GeM, IBN
from .batchrenorm import BatchRenormalization2D, BatchRenormalization2D_Noniid, BatchRenormalization1D


class EMA(nn.Module):
    def __init__(self, channels, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class EMABasicBlock(nn.Module):
    def __init__(self, block, dim, renorm, ibn, restride=False, non_iid=0):
        super(EMABasicBlock, self).__init__()
        if restride:
            block.conv1.stride = (1, 1)
            block.downsample[0].stride = (1, 1)
        if renorm:
            # experimental
            if non_iid:
                if not ibn:
                    block.bn1 = BatchRenormalization2D_Noniid(dim, non_iid, block.bn1.state_dict())
                block.bn2 = BatchRenormalization2D_Noniid(dim, non_iid, block.bn2.state_dict())
            else:
                if not ibn:
                    block.bn1 = BatchRenormalization2D(dim, block.bn1.state_dict())
                block.bn2 = BatchRenormalization2D(dim, block.bn2.state_dict())

        if ibn:
            # bn1 will be covered
            if renorm:
                if non_iid:
                    block.bn1.BN = BatchRenormalization2D_Noniid(dim >> 1, non_iid, block.bn1.BN.state_dict())
                else:
                    block.bn1.BN = BatchRenormalization2D(dim >> 1, block.bn1.BN.state_dict())

        # block.relu = AconC(dim)
        if list(block.named_children())[-1][0] == "downsample":
            self.block_pre = nn.Sequential(*list(block.children())[:-1])
            self.block_post = block.downsample
            if renorm:
                if non_iid:
                    self.block_post[1] = BatchRenormalization2D_Noniid(dim, non_iid, self.block_post[1].state_dict())
                else:
                    self.block_post[1] = BatchRenormalization2D(dim, self.block_post[1].state_dict())
        else:
            self.block_pre = block
            self.block_post = None
        self.emablock = EMA(dim)

    def forward(self, x):
        branch = x
        x = self.block_pre(x)
        x = self.emablock(x)
        if self.block_post:
            branch = self.block_post(branch)
        x += branch
        return F.relu(x)


class EMAPreActBasicBlock(EMABasicBlock):
    """I did not think of a proper pretrained weight to load
    """

    def __init__(self, block, dim, renorm, ibn, restride=False, non_iid=0):
        super(EMAPreActBasicBlock, self).__init__(block, dim, renorm, ibn, restride, non_iid)
        in_planes = block.conv1.in_channels
        if renorm:
            if non_iid:
                block.bn1 = BatchRenormalization2D_Noniid(in_planes, non_iid, block.bn1.state_dict())
            else:
                block.bn1 = BatchRenormalization2D(in_planes, block.bn1.state_dict())
        if ibn:
            block.bn1 = IBN(in_planes, renorm=renorm)
        self.block_pre = block

    def forward(self, x):
        branch = x
        x = self.block_pre[2](self.block_pre[1](x))
        if self.block_post:
            # This is tricky in onnx inference as you need to disable conv fusion
            branch = self.block_post[0](x)
        x = self.block_pre[0](x)
        x = self.block_pre[3](self.block_pre[2](self.block_pre[4](x)))
        x = self.emablock(x)
        x += branch
        return F.relu(x)


class EMARes18_IBN(nn.Module):
    """
    Additionally, we would like to test the network with local average pooling
    i.e. Divide into eight and concatenate them
    """

    def __init__(self,
                 num_class=751,
                 num_cams=6,
                 pooling="gem",
                 renorm=False,
                 is_reid=False,
                 non_iid=0):
        super().__init__()
        # model = models.resnet18(weights=resnet18_pretrained, progress=False)
        model = torch.hub.load("XingangPan/IBN-Net", "resnet18_ibn_a", pretrained=True)
        self.conv0 = model.conv1
        if renorm:
            if non_iid:
                self.bn0 = BatchRenormalization2D_Noniid(64, non_iid, model.bn1.state_dict())
            else:
                self.bn0 = BatchRenormalization2D(64, model.bn1.state_dict())
        else:
            self.bn0 = model.bn1
        self.relu0 = model.relu
        self.pooling0 = model.maxpool

        self.basicBlock11 = EMABasicBlock(model.layer1[0], 64, renorm, True, non_iid=non_iid)

        self.basicBlock12 = EMABasicBlock(model.layer1[1], 64, renorm, True, non_iid=non_iid)

        self.basicBlock21 = EMABasicBlock(model.layer2[0], 128, renorm, True, non_iid=non_iid)

        self.basicBlock22 = EMABasicBlock(model.layer2[1], 128, renorm, True, non_iid=non_iid)

        self.basicBlock31 = EMABasicBlock(model.layer3[0], 256, renorm, True, non_iid=non_iid)

        self.basicBlock32 = EMABasicBlock(model.layer3[1], 256, renorm, True, non_iid=non_iid)

        # last stride = 1
        self.basicBlock41 = EMABasicBlock(model.layer4[0], 512, renorm, False, True, non_iid=non_iid)

        self.basicBlock42 = EMABasicBlock(model.layer4[1], 512, renorm, False, non_iid=non_iid)

        if pooling == "gem":
            self.avgpooling = GeM()
        else:
            self.avgpooling = nn.AdaptiveAvgPool2d(1)

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
        trunc_normal_(self.cam_bias, std=0.02)
        self.cam_factor = 1.

    def forward(self, x, cam=None):
        x = self.conv0(x)
        x = self.bn0(x)
        # x = self.relu0(x) # probably you will get fewer dead neurons
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

        x_norm = self.bnneck(feature)
        if cam is not None:  # before or after bn?
            x_norm = x_norm + self.cam_factor * self.cam_bias[cam]
        x = self.classifier(x_norm)
        if self.is_reid:
            return x
        if not self.training:
            return x_norm, x
        return feature, x


def emares18_ibn(num_classes=751, loss="triplet", **kwargs):
    if loss == "triplet":
        is_reid = False
    elif loss == "softmax":
        is_reid = True
    else:
        raise NotImplementedError
    model = EMARes18_IBN(num_class=num_classes,
                         is_reid=is_reid,
                         **kwargs)
    return model
