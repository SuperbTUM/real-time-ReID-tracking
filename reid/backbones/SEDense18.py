import torch
import torch.nn as nn
from torchvision import models
from SERes18_IBN import SEBlock, GeM, IBN
from batchrenorm import BatchRenormalization2D
from attention_pooling import AttentionPooling


class SEDense18_IBN(nn.Module):
    """
    Additionally, we would like to test the network with local average pooling
    i.e. Divide into eight and concatenate them
    """
    def __init__(self,
                 resnet18_pretrained="IMAGENET1K_V1",
                 num_class=751,
                 needs_norm=True,
                 pooling="gem",
                 renorm=False,
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

        model.layer1[0].bn1 = IBN(64)
        self.basicBlock11 = model.layer1[0]
        if renorm:
            self.basicBlock11.bn2 = BatchRenormalization2D(64)
        self.seblock1 = SEBlock(64)

        self.basicBlock12 = model.layer1[1]
        if renorm:
            self.basicBlock12.bn1 = BatchRenormalization2D(64)
            self.basicBlock12.bn2 = BatchRenormalization2D(64)
        self.seblock2 = SEBlock(64)

        model.layer2[0].bn1 = IBN(128)
        self.basicBlock21 = model.layer2[0]
        if renorm:
            self.basicBlock21.bn2 = BatchRenormalization2D(128)
        self.seblock3 = SEBlock(128)
        self.ancillaryconv3 = nn.Conv2d(64, 128, 1, 2, 0)
        self.optionalNorm2dconv3 = nn.BatchNorm2d(128)

        self.basicBlock22 = model.layer2[1]
        if renorm:
            self.basicBlock22.bn1 = BatchRenormalization2D(128)
            self.basicBlock22.bn2 = BatchRenormalization2D(128)
        self.seblock4 = SEBlock(128)

        model.layer3[0].bn1 = IBN(256)
        self.basicBlock31 = model.layer3[0]
        if renorm:
            self.basicBlock31.bn2 = BatchRenormalization2D(256)
        self.seblock5 = SEBlock(256)
        self.ancillaryconv5 = nn.Conv2d(128, 256, 1, 2, 0)
        self.optionalNorm2dconv5 = nn.BatchNorm2d(256)

        self.basicBlock32 = model.layer3[1]
        if renorm:
            self.basicBlock32.bn1 = BatchRenormalization2D(256)
            self.basicBlock32.bn2 = BatchRenormalization2D(256)
        self.seblock6 = SEBlock(256)

        model.layer4[0].bn1 = IBN(512)
        self.basicBlock41 = model.layer4[0]
        if renorm:
            self.basicBlock41.bn2 = BatchRenormalization2D(512)
        # last stride = 1
        self.basicBlock41.conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.basicBlock41.downsample[0] = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.seblock7 = SEBlock(512)
        self.ancillaryconv7 = nn.Conv2d(256, 512, 1, 1, 0)
        self.optionalNorm2dconv7 = nn.BatchNorm2d(512)

        self.basicBlock42 = model.layer4[1]
        if renorm:
            self.basicBlock42.bn1 = BatchRenormalization2D(512)
            self.basicBlock42.bn2 = BatchRenormalization2D(512)
        self.seblock8 = SEBlock(512)

        if pooling == "gem":
            self.avgpooling = GeM()
        elif pooling == "attn":
            self.avgpooling = AttentionPooling(512)  # not working so well
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

        x = self.avgpooling(x)
        feature = x.view(x.size(0), -1)
        if self.is_reid:
            return feature
        x = self.bnneck(feature)
        x = self.classifier(x)

        return feature, x


def seden18_ibn(num_classes=751, pretrained="IMAGENET1K_V1", loss="triplet", **kwargs):
    if loss == "triplet":
        is_reid = False
    elif loss == "softmax":
        is_reid = True
    else:
        raise NotImplementedError
    model = SEDense18_IBN(num_class=num_classes,
                          resnet18_pretrained=pretrained,
                          is_reid=is_reid,
                          **kwargs)
    return model
