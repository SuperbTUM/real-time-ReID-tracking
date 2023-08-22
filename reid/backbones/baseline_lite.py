import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable

from .weight_init import weights_init_kaiming, weights_init_classifier


######################################################################

# Defines the new fc layer and classification layer
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, bnorm=True, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if bnorm:
            add_block += [nn.BatchNorm1d(input_dim)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x_normed = self.add_block(x)
        x_classified = self.classifier(x_normed)
        if self.return_f:
            return x, x_classified
        return x_classified


# Define the ResNet18-based Model
class ft_baseline(nn.Module):

    def __init__(self, class_num, stride=1):
        super(ft_baseline, self).__init__()
        model_ft = models.resnet18(weights="IMAGENET1K_V1")
        self.model = model_ft
        # avg pooling to global pooling
        if stride == 1:
            self.model.layer4[0].downsample[0].stride = (1, 1)
            self.model.layer4[0].conv1.stride = (1, 1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = ClassBlock(512, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        feature = x.view(x.size(0), x.size(1))
        x = self.classifier(feature)
        return feature, x
