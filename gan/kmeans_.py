from gan_utils import *
try:
    from faiss import Kmeans  # faiss should be faster
except ImportError or ModuleNotFoundError:
    from sklearn.cluster import KMeans

from torchvision import models
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_repres(reid_dataset):
    backbone = models.resnet50(weights="IMAGENET1K_V2").to(device)
    backbone = nn.Sequential(*list(backbone.children())[:-1])
    backbone.eval()
    backbone = torch.jit.script(backbone)  # torchscript
    data_loader = DataLoaderX(reid_dataset,
                              shuffle=False,
                              batch_size=1,
                              num_workers=1,
                              pin_memory=True)
    repres = []
    with torch.no_grad():
        for sample in tqdm(data_loader):
            img, label = sample
            img = img.to(device)
            repre = backbone(img).squeeze().cpu().numpy()
            repres.append(repre)
    repres = np.asarray(repres)
    return repres


def get_labels(repres, n_clusters=2):
    try:
        kmeans = Kmeans(d=repres.shape[1], k=n_clusters, niter=300, nredo=10)
        kmeans.train(repres.astype(np.float32))
        return kmeans.index.search(repres.astype(np.float32), 1)[1].flatten()
    except:
        kmeans = KMeans(n_clusters, random_state=0)
        return kmeans.fit_predict(repres)


def get_groups(reid_dataset, k=2):
    repres = get_repres(reid_dataset)
    labels = get_labels(repres, k)
    return labels
