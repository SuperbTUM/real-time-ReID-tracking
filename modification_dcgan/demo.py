__author__ = "Mingzhe Hu"

import glob

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from prefetch_generator import BackgroundGenerator
from scipy.io import loadmat
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def load_appearence(path):
    app_info = loadmat(path)
    return app_info["gallery"]


def fetch_query(path):
    query = glob.glob(path + "*.jpg")
    query = list(map(lambda x: x.split("\\")[-1], query))
    query = sorted(query)
    return np.asarray(query)


def fetch_gallery(path):
    gallery = glob.glob(path + "*.jpg")
    gallery = list(map(lambda x: x.split("\\")[-1], gallery))
    gallery = list(filter(lambda x: x[:4] != "0000" and x[0] != "-", gallery))
    gallery = sorted(gallery)
    return np.asarray(gallery)


def construct_raw_dataset(query_images):
    labels = np.empty((len(query_images), ), dtype=np.int32)
    cnt = 0
    cur = 1
    for i in range(len(query_images)):
        image_info = query_images[i][:4]
        id = int(image_info)
        if id != cur:
            cur = id
            cnt += 1
        labels[i] = cnt
    raw_dataset = list(zip(query_images, labels))
    return np.asarray(raw_dataset), cnt + 2  # number of classes


class DataSet4GAN(Dataset):
    def __init__(self, raw_dataset, transform=None):
        super(DataSet4GAN, self).__init__()
        self.raw_dataset = raw_dataset
        self.transform = transform

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, item):
        image_path, label = self.raw_dataset[item]
        img_data = Image.open("Market1501\\query/" + image_path)
        if self.transform:
            img_data = self.transform(img_data)
        label = torch.tensor(int(label)).int()
        return img_data, label


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu=1, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu=1, nc=3, ndf=64, num_classes=751):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, num_classes, 4, 1, 0, bias=False),
            # nn.Softmax(dim=1)
        )

    def forward(self, input):
        return self.main(input)


def load_model(num_classes, device="cuda:0", lr=1e-3):
    modelG = Generator().to(device)
    modelG.apply(weights_init)
    modelD = Discriminator(num_classes=num_classes).to(device)
    modelD.apply(weights_init)
    criterion = nn.CrossEntropyLoss()
    optimizerG = torch.optim.Adam(modelG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(modelD.parameters(), lr=lr, betas=(0.5, 0.999))
    return modelG, modelD, criterion, optimizerG, optimizerD


def load_dataset(raw_dataset, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.CenterCrop((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    reid_dataset = DataSet4GAN(raw_dataset, transform)
    data_loader = DataLoaderX(reid_dataset,
                              shuffle=True,
                              batch_size=batch_size,
                              num_workers=4,
                              pin_memory=True)
    return data_loader


def train(raw_dataset, epochs=20, batch_size=32, nz=100, device="cuda:0", num_classes=751):
    modelG, modelD, criterion, optimizerG, optimizerD = load_model(num_classes=num_classes, device=device)
    G_losses = []
    D_losses = []
    iters = 0
    for epoch in range(epochs):
        dataloader = load_dataset(raw_dataset, batch_size=batch_size)
        i = 0
        for sample in tqdm(dataloader):
            modelD.zero_grad()
            cur_image, label = sample
            cur_image = cur_image.to(device)
            label = label.long()
            label = label.to(device)
            label_copy = label.clone()
            classification = modelD(cur_image)
            loss = criterion(classification.squeeze(), label)
            loss.backward()
            optimizerD.step()
            D_x = classification.mean().item()

            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            generated = modelG(noise)
            label.fill_(num_classes-1)
            which_person = modelD(generated.detach())
            loss_generated_discriminate = criterion(which_person.squeeze(), label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            loss_generated_discriminate.backward()
            D_G_z1 = which_person.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = loss + loss_generated_discriminate
            # Update D
            optimizerD.step()

            modelG.zero_grad()
            label = label_copy
            confuse = modelD(generated)
            errG = criterion(confuse.squeeze(), label)
            errG.backward()
            D_G_z2 = confuse.mean().item()
            # Update G
            optimizerG.step()

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            i += 1
            if i % 50 == 0:
                print("epoch: {:d}, Loss G: {:.5f}, Loss D: {:.5f}".format(epoch, errG.item(), errD.item()))
            # Check how the generator is doing by saving G's output on fixed_noise
            iters += 1

    return G_losses, D_losses


def plot(G_losses, D_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    query_images = fetch_query("Market1501\\query/")
    raw_dataset, num_classes = construct_raw_dataset(query_images)
    G_losses, D_losses = train(raw_dataset, device="cpu", num_classes=num_classes)
    plot(G_losses, D_losses)
