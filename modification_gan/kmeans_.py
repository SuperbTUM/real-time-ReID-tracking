from synthetic_generate import DataSet4GAN, fetch_rawdata, construct_raw_dataset, DataLoaderX
try:
    from faiss import Kmeans  # faiss should be faster
except ImportError:
    from sklearn.cluster import KMeans

from torchvision import transforms, models
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def get_repres(query_images):
    raw_dataset, num_classes = construct_raw_dataset(query_images)
    transform = transforms.Compose([
        transforms.Resize((128, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # We will split them to 2 sub-datasets

    query_images = fetch_rawdata("Market1501/bounding_box_train/", "Market1501/bounding_box_test/")
    repres = get_repres(query_images)
    labels = get_labels(repres)
