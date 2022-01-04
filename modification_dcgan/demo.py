__author__ = "Mingzhe Hu"

import glob

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from prefetch_generator import BackgroundGenerator
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import os
import matplotlib.animation as animation
from IPython.display import HTML


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
    labels = np.empty((len(query_images),), dtype=np.int32)
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
            nn.ConvTranspose2d(ngf, nc, (8, 4), 2, 1, bias=False),
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
            nn.Conv2d(nc, ndf, (8, 4), 2, 1, bias=False),
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


def load_model(nz, device="cuda:0", lr=0.0002):
    modelG = Generator(nz=nz).to(device)
    modelG.apply(weights_init)
    modelD = Discriminator().to(device)
    modelD.apply(weights_init)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()
    # criterion = LabelSmoothing()
    optimizerG = torch.optim.Adam(modelG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(modelD.parameters(), lr=lr, betas=(0.5, 0.999))
    lr_schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, 1000, 0.5)
    lr_schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, 1000, 0.75)
    return modelG, modelD, criterion, optimizerG, optimizerD, lr_schedulerG, lr_schedulerD


def load_dataset(raw_dataset, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    reid_dataset = DataSet4GAN(raw_dataset, transform)
    data_loader = DataLoaderX(reid_dataset,
                              shuffle=True,
                              batch_size=batch_size,
                              num_workers=16,
                              pin_memory=True)
    return data_loader


def load_everything(nz=100, device="cuda:0", lr=1e-2):
    print("-----------------------------Loading Generator and Discriminator-----------------------------------")
    modelG, modelD, criterion, optimizerG, optimizerD, lr_schedulerG, lr_schedulerD = load_model(nz, device=device,
                                                                                                 lr=lr)
    emaG = EMA(modelG, 0.999)
    emaG.register()
    return modelG, modelD, criterion, optimizerG, optimizerD, lr_schedulerG, lr_schedulerD, emaG


def train(raw_dataset,
          checkpoint=None,
          epochs=10,
          batch_size=64,
          nz=100,
          device="cuda:0",
          lr=1e-3,
          training=True):
    netG, netD, criterion, optimizerG, optimizerD, lr_schedulerG, lr_schedulerD, emaG = load_everything(nz, device, lr)
    # Training Loop
    if not training:
        return netG, checkpoint
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    real_label = 1.
    fake_label = 0.
    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(epochs):
        # For each batch in the dataloader
        dataloader = load_dataset(raw_dataset, batch_size)
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == 4) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            iters += 1
    torch.save(netG.state_dict(), "Generate_model_trained.t7")
    torch.cuda.empty_cache()
    return G_losses, D_losses, netG, img_list


def plot(G_losses, D_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_imglist(img_list):
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())


def generate(modelG_checkpoint, modelG, device="cuda:0"):
    # emaG.apply_shadow()
    print("-----------------------------------Start Generating New Image-------------------------------------")
    width, height = 64, 128
    if os.path.exists(modelG_checkpoint):
        modelG.load_state_dict(torch.load(modelG_checkpoint), strict=False)
        modelG.eval()
    with torch.no_grad():
        fixed_noise = torch.randn(1, 100, 1, 1).to(device)
        generated_image = modelG(fixed_noise)
    # emaG.restore()

    generated_image = generated_image.squeeze()
    generated_image = transforms.Resize((width, height))(generated_image)
    generated_image = generated_image.cpu().detach().numpy().transpose(1, 2, 0)
    return generated_image


if __name__ == "__main__":
    query_images = fetch_rawdata("Market1501\\bounding_box_train\\", "Market1501\\bounding_box_test\\")
    raw_dataset, num_classes = construct_raw_dataset(query_images)
    G_losses, D_losses, netG, img_list = train(raw_dataset, 20)
    plot(G_losses, D_losses)
    plot_imglist(img_list)
