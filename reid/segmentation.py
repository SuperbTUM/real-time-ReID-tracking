# experimental
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from video_reid_train import TripletLoss, CenterLoss

import glob
from reid.backbones.osnet import osnet_ibn_x1_0

# use deeplabv3_resnet50 instead of deeplabv3_resnet101 to reduce the model size
model = torch.hub.load('pytorch/vision:v0.8.0', 'deeplabv3_resnet50', pretrained=True)
model.eval()
scriptedm = torch.jit.script(model)

# There are two ways, one is to segment on blured image, one is on original image
def inference(input_image, blured=False):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    blur = transforms.Compose([
        transforms.GaussianBlur(7),
    ])
    input_tensor = preprocess(input_image)
    if blured:
        input_tensor = blur(input_tensor)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = scriptedm(input_batch)['out'][0]  #(21, W, H), 21 means classes
        output = output.argmax(0)
    return output.cpu().numpy()


def extract_foreground_background(output, input_image):
    input_image = np.array(input_image)
    foreground = deepcopy(input_image)
    background = deepcopy(input_image)
    for h in range(output.shape[0]):
        for w in range(output.shape[1]):
            if output[h, w] == 0:
                foreground[h, w, :] = 0
            else:
                background[h, w, :] = 0
    return foreground, background


def batched_extraction(working_dir, blured=False):
    images_path = glob.glob("/".join((working_dir, "*.jpg")))
    foreground, background = [], []
    for path in images_path:
        img = Image.open(path)
        output = inference(img, blured)
        f, b = extract_foreground_background(output, img)
        foreground.append(f)
        background.append(b)
    return foreground, background


class ReIDModel(nn.Module):
    def __init__(self, pretrained=True, num_class=751):
        super(ReIDModel, self).__init__()
        base_model = osnet_ibn_x1_0(pretrained=pretrained, num_classes=num_class)
        self.osnet = nn.Sequential(*list(base_model.children())[:-2])
        self.fc1 = self._construct_fc_layer((512, ), 512, dropout_p=None)
        self.fc2 = self._construct_fc_layer((512, ), 512, dropout_p=None)
        self.classifier = nn.Linear(1024, num_class)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None

        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def forward(self, img1, img2):
        foreground_embedding = self.fc1(self.osnet(img1))
        background_embedding = self.fc2(self.osnet(img2))
        pred = self.classifier(torch.cat((foreground_embedding, background_embedding), dim=-1))
        return foreground_embedding, background_embedding, pred


class ExtractedDataset(Dataset):
    def __init__(self, foregrounds, backgrounds, labels, transform=None):
        super(ExtractedDataset, self).__init__()
        self.foregrounds = foregrounds
        self.backgrounds = backgrounds
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.foregrounds)

    def __getitem__(self, item):
        foreground_img = self.foregrounds[item]
        background_img = self.backgrounds[item]
        label = self.labels[item]
        if not torch.is_tensor(label):
            label = torch.tensor(label)
        if isinstance(foreground_img, np.ndarray):
            foreground_img = Image.fromarray(foreground_img)
        if isinstance(background_img, np.ndarray):
            background_img = Image.fromarray(background_img)
        if self.transform:
            foreground_img = self.transform(foreground_img)
            background_img = self.transform(background_img)
        return foreground_img, background_img, label


class MixedCE(nn.Module):
    def __init__(self, alpha=0.5):
        super(MixedCE, self).__init__()
        self.alpha = alpha

    def forward(self, foreground_embedding, background_embedding, label):
        foreground = torch.logsumexp(foreground_embedding, dim=-1)
        background = torch.logsumexp(background_embedding, dim=-1)
        loss_foreground = foreground - torch.gather(foreground_embedding, 1, label[:, None].expand(-1, foreground_embedding.size(1)))[:, 0]
        loss_foreground = loss_foreground.sum()
        loss_background = background - torch.gather(background_embedding, 1, label[:, None].expand(-1, background_embedding.size(1)))[:, 0]
        loss_background = loss_background.sum()
        return self.alpha * loss_foreground + (1 - self.alpha) * loss_background


class ReID_Loss(nn.Module):
    def __init__(self, alpha, lamda1, lamda2, num_class):
        super(ReID_Loss, self).__init__()
        self.mixedCE = MixedCE(alpha)
        self.triplet = TripletLoss()
        self.center_loss = CenterLoss(num_classes=num_class)
        self.lamda1 = lamda1
        self.lamda2 = lamda2

    def forward(self, foreground_embedding, background_embedding, label):
        mixed_ce = self.mixedCE(foreground_embedding, background_embedding, label)
        triplet = self.triplet(torch.cat((foreground_embedding, background_embedding), 1), label)
        center = self.center_loss(torch.cat((foreground_embedding, background_embedding), 1), label)
        return (1 - self.lamda1 - self.lamda2) * mixed_ce + self.lamda1 * triplet + self.lamda2 * center


if __name__ == "__main__":
    pass
