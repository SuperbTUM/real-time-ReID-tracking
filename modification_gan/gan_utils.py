import glob
from PIL import Image
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
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
    return np.asarray(total_images)


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
    return np.asarray(raw_dataset), cnt + 1  # number of classes


class DataSet4GAN(Dataset):
    def __init__(self, raw_dataset, transform=None, group=-1):
        super(DataSet4GAN, self).__init__()
        if group >= 0:
            raw_dataset = filter(lambda x: x[-1] == group, raw_dataset)
        self.raw_dataset = raw_dataset
        self.transform = transform

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, item):
        image_path, label = self.raw_dataset[item][:2]
        img_data = Image.open(image_path).convert("RGB")
        if self.transform:
            img_data = self.transform(img_data)
        label = torch.tensor(int(label)).float()
        return img_data, label
