import glob
from PIL import Image
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, Dataset


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def fetch_rawdata(*paths):
    total_images = list()
    for path in paths:
        query = glob.glob(path + "*.jpg")
        query = list(map(lambda x: "/".join(x.split("/")[-3:]), query))
        query = list(filter(lambda x: x.split("/")[-1][:4] != "0000" or x.split("/")[-1][0] != "-", query))
        total_images.extend(query)
    return total_images


def construct_raw_dataset(query_images):
    query_images = list(filter(lambda x: x.split("/")[2][0] != "-", query_images))
    labels = np.empty((len(query_images),), dtype=np.int32)
    cnt = 0
    cur = 1
    for i in range(len(query_images)):
        image_info = query_images[i].split("/")[2][:4]
        id = int(image_info)
        if id != cur:
            cur = id
            cnt += 1
        labels[i] = cnt
    raw_dataset = list(zip(query_images, labels))
    return raw_dataset, cnt + 1  # number of classes


class DataSet4GAN(Dataset):
    def __init__(self, raw_dataset, root, transform=None, group=-1):
        super(DataSet4GAN, self).__init__()
        if group > 0:
            raw_dataset = list(filter(lambda x: x[-1] == group, raw_dataset))
        self.raw_dataset = raw_dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, item):
        image_path, label = self.raw_dataset[item][:2]
        image_path = "/".join((self.root, image_path))
        img_data = Image.open(image_path).convert("RGB")
        if self.transform:
            img_data = self.transform(img_data)
        label = torch.tensor(int(label)).float()
        return img_data, label


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def check_parameters(model):
    # credit to https://discuss.pytorch.org/t/finding-model-size/130275
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb
