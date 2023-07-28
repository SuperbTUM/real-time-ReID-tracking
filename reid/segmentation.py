# experimental
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


# use deeplabv3_resnet50 instead of deeplabv3_resnet101 to reduce the model size
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', weights="DEFAULT")
model.eval()
scriptedm = torch.jit.script(model).cuda()


# There are two ways, one is to segment on blured image, one is on original image
def inference(scriptedm, input_image, blured=False):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    blur = nn.Sequential(transforms.GaussianBlur(7))
    blurts = torch.jit.script(blur)
    input_tensor = preprocess(input_image)
    if blured:
        input_tensor = blurts(input_tensor)
    input_batch = input_tensor.unsqueeze(0).cuda()
    with torch.no_grad():
        output = scriptedm(input_batch)['out'][0]  #(21, W, H), 21 means classes
        output = output.argmax(0)
    return output.cpu().numpy()


def extract_foreground_background(output, input_image, blur_background=True):
    input_image_array = np.array(input_image)
    foreground = deepcopy(input_image_array)
    background = deepcopy(input_image)
    blur = transforms.GaussianBlur(7)
    if blur_background:
        background = blur(background)
        background = np.array(background)
    for h in range(output.shape[0]):
        for w in range(output.shape[1]):
            if output[h, w] == 0:
                foreground[h, w, :] = 0
            else:
                background[h, w, :] = 0
    return foreground, background


def batched_extraction(images_path, blured=False, blur_background=True):
    foreground, background = [], []
    for path in images_path:
        if isinstance(path, str):
            img = Image.open(path).convert("RGB")
        else:
            img = path
        output = inference(scriptedm, img, blured)
        f, b = extract_foreground_background(output, img, blur_background)
        foreground.append(f)
        background.append(b)
    return foreground, background


class ExtractedDataset(Dataset):
    def __init__(self, images_path, transform=None, blured=True):
        super(ExtractedDataset, self).__init__()
        paths = list(map(lambda x: x[0], images_path))
        self.foregrounds, self.backgrounds = batched_extraction(paths, blured)
        self.transform = transform

        self.images_path = images_path

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        foreground_img = self.foregrounds[item]
        background_img = self.backgrounds[item]
        full_img = foreground_img + background_img

        label, cam, seq = self.images_path[item][1:]
        if not torch.is_tensor(label):
            label = torch.tensor(label)
        if not torch.is_tensor(cam):
            label = torch.tensor(cam)
        if not torch.is_tensor(seq):
            label = torch.tensor(seq)
        # if isinstance(foreground_img, np.ndarray):
        #     foreground_img = Image.fromarray(foreground_img)
        # if isinstance(background_img, np.ndarray):
        #     background_img = Image.fromarray(background_img)
        if isinstance(full_img, np.ndarray):
            full_img = Image.fromarray(full_img)
        if self.transform:
            # foreground_img = self.transform(foreground_img)
            # background_img = self.transform(background_img)
            full_img = self.transform(full_img)
        return full_img, label, cam, seq


if __name__ == "__main__":
    pass
