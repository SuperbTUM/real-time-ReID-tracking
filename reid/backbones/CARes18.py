import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .SERes18_IBN import GeM, IBN, trunc_normal_, weights_init_classifier, weights_init_kaiming
from .batchrenorm import BatchRenormalization2D, BatchRenormalization2D_Noniid, BatchRenormalization1D


class CABlock(nn.Module):
    def __init__(self, channel, reduction=16, renorm=False, non_iid=0):
        super(CABlock, self).__init__()

        # self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)
        self.conv_1x1 = nn.Linear(channel, channel//reduction, bias=False)  # bias=False

        self.relu = nn.ReLU(inplace=True)
        if renorm:
            if non_iid:
                self.bn = BatchRenormalization2D_Noniid(channel//reduction, non_iid)
            else:
                self.bn = BatchRenormalization2D(channel//reduction)
        else:
            self.bn = nn.BatchNorm2d(channel//reduction)

        # self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        # self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_h = nn.Linear(channel//reduction, channel, bias=False)  # bias=False
        self.F_w = nn.Linear(channel//reduction, channel, bias=False)  # bias=False

        self.sigmoid = nn.Sigmoid()

        torch.nn.init.kaiming_normal_(self.conv_1x1.weight.data, a=0, mode='fan_out')
        # torch.nn.init.constant_(self.conv_1x1.bias.data, 0.0)
        torch.nn.init.kaiming_normal_(self.F_h.weight.data, a=0, mode='fan_out')
        # torch.nn.init.constant_(self.F_h.bias.data, 0.0)
        torch.nn.init.kaiming_normal_(self.F_w.weight.data, a=0, mode='fan_out')
        # torch.nn.init.constant_(self.F_w.bias.data, 0.0)

    def forward(self, x):
        h, w = x.size(2), x.size(3)

        x_h = F.adaptive_avg_pool2d(x, (h, 1)).permute(0, 1, 3, 2)
        x_w = F.adaptive_avg_pool2d(x, (1, w))
        concat = torch.cat((x_h, x_w), 3).permute(0, 2, 3, 1) # (32, 64, 1, 96) -> (32, 1, 96, 64)

        # x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(concat)))
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(concat).permute(0, 3, 1, 2))) # (32, 4, 1, 96)

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        # s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        # s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        s_h = self.sigmoid(self.F_h(x_cat_conv_split_h.permute(0, 3, 2, 1)).permute(0, 3, 1, 2)) # (32, 64, 1, 64) -> (32, 64, 64, 1)
        s_w = self.sigmoid(self.F_w(x_cat_conv_split_w.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)) # (32, 32, 4, 1) -> (32, 64, 1, 32)

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out


# class CASEBlock(CABlock):
#     def __init__(self, channel, reduction=16, renorm=False, non_iid=0):
#         super(CASEBlock, self).__init__(channel, reduction, renorm, non_iid)
#         self.F_c = nn.Linear(channel // reduction, channel, bias=False)  # bias=False
#         torch.nn.init.kaiming_normal_(self.F_c.weight.data, a=0, mode='fan_out')
#
#     def forward(self, x):
#         h, w = x.size(2), x.size(3)
#
#         x_h = F.adaptive_avg_pool2d(x, (h, 1)).permute(0, 1, 3, 2)
#         x_w = F.adaptive_avg_pool2d(x, (1, w))
#         x_c = F.adaptive_avg_pool2d(x, 1)
#         concat = torch.cat((x_h, x_w, x_c), 3).permute(0, 2, 3, 1)  # (32, 64, 1, 97) ->
#
#         # x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(concat)))
#         x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(concat).permute(0, 3, 1, 2)))  # (32, 4, 1, 97)
#
#         x_cat_conv_split_h, x_cat_conv_split_w, x_cat_conv_split_c = x_cat_conv_relu.split([h, w, 1], 3)
#
#         # s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
#         # s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
#         s_h = self.sigmoid(
#             self.F_h(x_cat_conv_split_h.permute(0, 3, 2, 1)).permute(0, 3, 1, 2))  # (32, 64, 1, 64) -> (32, 64, 64, 1)
#         s_w = self.sigmoid(
#             self.F_w(x_cat_conv_split_w.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))  # (32, 32, 4, 1) -> (32, 64, 1, 32)
#         s_c = self.sigmoid(
#             self.F_c(x_cat_conv_split_c.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
#
#         out = x * s_h.expand_as(x) * s_w.expand_as(x) * s_c.expand_as(x)
#
#         return out


class CABasicBlock(nn.Module):
    def __init__(self, block, dim, renorm, ibn, restride=False, non_iid=0):
        super(CABasicBlock, self).__init__()
        if restride:
            block.conv1.stride = (1, 1)
            block.downsample[0].stride = (1, 1)
        # if ibn:
        #     pretrained_in = block.bn1.IN
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
            #     block.bn1 = IBN(dim, renorm=renorm, non_iid=non_iid)
            #     block.bn1.IN = pretrained_in
        # block.relu = AconC(dim)
        if list(block.named_children())[-1][0] == "downsample":
            self.block_pre = nn.Sequential(*list(block.children())[:-1])
            self.block_post = block.downsample
            if renorm:
                if non_iid:
                    self.block_post[1] = BatchRenormalization2D_Noniid(2 * dim, non_iid, self.block_post[1].state_dict())
                else:
                    self.block_post[1] = BatchRenormalization2D(2 * dim, self.block_post[1].state_dict())
        else:
            self.block_pre = block
            self.block_post = None
        self.cablock = CABlock(dim, renorm=renorm, non_iid=non_iid)

    def forward(self, x):
        branch = x
        x = self.block_pre(x)
        x = self.cablock(x)
        if self.block_post:
            branch = self.block_post(branch)
        x += branch
        return F.relu(x)


class CAPreActBasicBlock(CABasicBlock):
    """I did not think of a proper pretrained weight to load
    """
    def __init__(self, block, dim, renorm, ibn, restride=False, non_iid=0):
        super(CAPreActBasicBlock, self).__init__(block, dim, renorm, ibn, restride, non_iid)
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
        x = self.cablock(x)
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
                 needs_norm=False,
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

        self.basicBlock11 = CABasicBlock(model.layer1[0], 64, renorm, True, non_iid=non_iid)

        self.basicBlock12 = CABasicBlock(model.layer1[1], 64, renorm, True, non_iid=non_iid)

        self.basicBlock21 = CABasicBlock(model.layer2[0], 128, renorm, True, non_iid=non_iid)

        self.basicBlock22 = CABasicBlock(model.layer2[1], 128, renorm, True, non_iid=non_iid)

        self.basicBlock31 = CABasicBlock(model.layer3[0], 256, renorm, True, non_iid=non_iid)

        self.basicBlock32 = CABasicBlock(model.layer3[1], 256, renorm, True, non_iid=non_iid)

        # last stride = 1
        self.basicBlock41 = CABasicBlock(model.layer4[0], 512, renorm, False, True, non_iid=non_iid)

        self.basicBlock42 = CABasicBlock(model.layer4[1], 512, renorm, False, non_iid=non_iid)

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
        self.needs_norm = needs_norm
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
            feature = feature + self.cam_factor * self.cam_bias[cam]
            trunc_normal_(feature, std=0.02)
        if self.is_reid:
            return feature
        x_norm = self.bnneck(feature)
        x = self.classifier(x_norm)
        if self.needs_norm:
            return x_norm, x
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
