from synthetic_generate import DataSet4GAN, fetch_rawdata, construct_raw_dataset, DataLoaderX
from sklearn.cluster import KMeans
from torchvision import transforms, models
import torch
import torch.nn as nn
import numpy as np


def get_labels(repres, n_clusters=2):
    kmeans = KMeans(n_clusters, random_state=0)
    return kmeans.fit_transform(repres)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # We will split them to 2 sub-datasets

    query_images = fetch_rawdata("Market1501/bounding_box_train/", "Market1501/bounding_box_test/")
    raw_dataset, num_classes = construct_raw_dataset(query_images)
    transform = transforms.Compose([
        transforms.Resize((128, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    reid_dataset = DataSet4GAN(raw_dataset, transform)
    backbone = models.resnet50(pretrained=True)
    backbone = nn.Sequential(*list(backbone.children())[:-1]).to(device)
    backbone.eval()
    data_loader = DataLoaderX(reid_dataset,
                              shuffle=False,
                              batch_size=1,
                              num_workers=1,
                              pin_memory=True)
    repres = []
    with torch.no_grad():
        for sample in data_loader:
            sample = sample.to(device)
            repre = backbone(sample).squeeze().cpu().numpy()
            repres.append(repre)
    repres = np.asarray(repres)

    labels = get_labels(repres)
