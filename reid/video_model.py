import torch.nn.functional as F
from SERes18_IBN import SEDense18_IBN


# It's not working well as expected
class InVideoModel(SEDense18_IBN):
    def __init__(self, num_class, needs_norm=True, gem=True, is_reid=False):
        super(InVideoModel, self).__init__(num_class=num_class,
                                           needs_norm=needs_norm,
                                           gem=gem,
                                           is_reid=is_reid)

    def forward(self, x):
        b, s, c, h, w = x.size()
        x = x.view(b * s, c, h, w)
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

        bs, c, h, w = x.size()
        x = x.view(b, s, c, h, w).permute(0, 2, 1, 3, 4).reshape(b, c, s * h, w)

        x = self.avgpooling(x)
        feature = x.view(x.size(0), -1)
        if self.is_reid:
            # do we need a further normalization here?
            return F.normalize(feature, p=2, dim=1)
        x = self.bnneck(feature)
        x = self.classifier(x)

        return x, feature
