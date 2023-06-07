__author__ = "Mingzhe Hu"

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import matplotlib.animation as animation
from IPython.display import HTML
import argparse
from tqdm import tqdm

from gan_utils import *
from kmeans_ import get_groups
from .backbones.discrim import Discriminator
from .backbones.vae_gan import VAE, Generator, weights_init

device = "cuda"


def load_model(nz,
               device="cuda:0",
               lr=0.0002,
               spectral_norm=True,
               self_attn=False):
    modelG = Generator(nz=nz, spectral_norm=spectral_norm, self_attn=self_attn).to(device)
    modelG.apply(weights_init)
    modelD = Discriminator(spectral_norm=spectral_norm, self_attn=self_attn).to(device)
    modelD.apply(weights_init)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()
    # criterion = LabelSmoothing()
    optimizerG = torch.optim.Adam(modelG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(modelD.parameters(), lr=lr, betas=(0.5, 0.999))
    lr_schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, 1000, 0.5)
    lr_schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, 1000, 0.75)
    return modelG, modelD, criterion, optimizerG, optimizerD, lr_schedulerG, lr_schedulerD


def load_dataset(raw_dataset, batch_size=32, group=-1):
    transform = transforms.Compose([
        transforms.Resize((128, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    reid_dataset = DataSet4GAN(raw_dataset, params.root, transform, group=group)
    data_loader = DataLoaderX(reid_dataset,
                              shuffle=True,
                              batch_size=batch_size,
                              num_workers=4,
                              pin_memory=True)
    return data_loader


def load_everything(nz=100, device="cuda:0", lr=1e-2, spectral_norm=True, self_attn=False):
    print("-----------------------------Loading Generator and Discriminator-----------------------------------")
    modelG, modelD, criterion, optimizerG, optimizerD, lr_schedulerG, lr_schedulerD = load_model(nz,
                                                                                                 device=device,
                                                                                                 lr=lr,
                                                                                                 spectral_norm=spectral_norm,
                                                                                                 self_attn=self_attn)
    emaG = EMA(modelG, 0.999)
    emaG.register()
    return modelG, modelD, criterion, optimizerG, optimizerD, lr_schedulerG, lr_schedulerD, emaG


def VAE_GAN_train_one_epoch(iterator,
                            data,
                            gen,
                            discrim,
                            criterion,
                            optim_Dis,
                            optim_E,
                            optim_D,
                            Wassertein,
                            gp=False,
                            gamma=15,
                            lamda=10):
    bs, width, height = data.size(0), data.size(3), data.size(2)

    ones_label = torch.ones((bs, ), device=device)
    zeros_label = torch.zeros((bs, ), device=device)
    zeros_label1 = torch.zeros((bs, ), device=device)
    datav = data.to(device)
    mean, logvar, rec_enc = gen(datav)
    z_p = torch.randn(bs, 128, device=device)
    x_p_tilda = gen.decoder(z_p)

    output1 = discrim(datav)[0]
    if Wassertein:
        errD_real = -torch.mean(output1.squeeze())
    else:
        errD_real = criterion(output1.squeeze(), ones_label)
    output2 = discrim(x_p_tilda)[0]
    if Wassertein:
        errD_rec_noise = torch.mean(output2.squeeze())
    else:
        errD_rec_noise = criterion(output2.squeeze(), zeros_label1)
    if gp:
        alpha = torch.rand(bs, 1, 1, 1, device=device)
        x_hat = alpha * datav + (1 - alpha) * x_p_tilda
        output3 = discrim(x_hat)[0]
        gradients = torch.autograd.grad(outputs=output3, inputs=x_hat,
                                        grad_outputs=torch.ones_like(output3, device=device),
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        gradient_penalty = (gradients.view(gradients.size(0), -1).norm(2, 1) - 1) ** 2
        gradient_penalty = gradient_penalty.mean()
        dis_loss = errD_real + errD_rec_noise + lamda * gradient_penalty
    else:
        output3 = discrim(rec_enc)[0]
        dis_loss = errD_real + errD_rec_noise + torch.mean(output3.squeeze())
    optim_Dis.zero_grad()
    if dis_loss.item() < 0:
        print("Warning! Dis loss is below 0!")
    dis_loss.backward(retain_graph=True)
    optim_Dis.step()

    output4 = discrim(datav)[0]
    if Wassertein:
        errD_real = -torch.mean(output4.squeeze(1))  # ??
    else:
        errD_real = criterion(output4.squeeze(), ones_label)
    output5 = discrim(rec_enc)[0]
    if Wassertein:
        errD_rec_enc = torch.mean(output5.squeeze(1))
    else:
        errD_rec_enc = criterion(output5.squeeze(), zeros_label)
    output6 = discrim(x_p_tilda)[0]
    if Wassertein:
        errD_rec_noise = torch.mean(output6.squeeze(1))
    else:
        errD_rec_noise = criterion(output6.squeeze(), zeros_label1)
    gan_loss = errD_real + errD_rec_enc + errD_rec_noise

    x_l_tilda = discrim(rec_enc)[1]
    x_l = discrim(datav)[1]
    rec_loss = ((x_l_tilda - x_l) ** 2).mean()
    err_dec = gamma * rec_loss - gan_loss
    optim_D.zero_grad()
    if err_dec.item() < 0:
        print("Warning! Decoder loss is below 0!")
    err_dec.backward(retain_graph=True)
    optim_D.step()

    mean, logvar, rec_enc = gen(datav)
    x_l_tilda = discrim(rec_enc)[1]
    x_l = discrim(datav)[1]
    rec_loss = ((x_l_tilda - x_l) ** 2).mean()
    prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
    prior_loss = (-0.5 * torch.sum(prior_loss)) / torch.numel(mean.data)
    err_enc = prior_loss + 5 * rec_loss

    optim_E.zero_grad()
    if err_enc.item() < 0:
        print("Warning! Encoder loss is below 0!")
    err_enc.backward(retain_graph=True)
    optim_E.step()
    desc = "discriminator loss: {:.4f}, encoder loss: {:.4f}, decoder loss: {:.4f}".format(
        dis_loss.detach().cpu().item(),
        err_enc.detach().cpu().item(),
        err_dec.detach().cpu().item())
    iterator.set_description(desc=desc)


def train_VAE_GAN(raw_dataset,
                  checkpoint=None,
                  epochs=10,
                  batch_size=64,
                  device="cuda:0",
                  lr=1e-3,
                  gamma=20,
                  training=True,
                  Wassertein=False,
                  gp=False,
                  threshold=0.05,
                  spectral_norm=True,
                  self_attn=False
                    ):
    if not training:
        return checkpoint
    gen = VAE(spectral_norm=spectral_norm, self_attn=self_attn).to(device)
    discrim = Discriminator(VAE=True, Wassertein=Wassertein, spectral_norm=spectral_norm, self_attn=self_attn).to(device)
    criterion = nn.BCELoss().to(device)
    optim_E = torch.optim.RMSprop(gen.encoder.parameters(), lr=lr, alpha=0.9, eps=1e-8, centered=False)
    optim_D = torch.optim.RMSprop(gen.decoder.parameters(), lr=lr, alpha=0.9, eps=1e-8, centered=False)
    optim_Dis = torch.optim.RMSprop(discrim.parameters(), lr=lr, alpha=0.9, eps=1e-8, centered=False)
    scheduler_E = torch.optim.lr_scheduler.ExponentialLR(optim_E, gamma=0.75)
    scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optim_D, gamma=0.75)
    scheduler_Dis = torch.optim.lr_scheduler.ExponentialLR(optim_Dis, gamma=0.75)

    lamda = 10
    dataloader = load_dataset(raw_dataset, batch_size)
    for epoch in range(epochs):
        if epoch % 5 == 0:
            lamda -= 1
        iterator = tqdm(dataloader)
        for data in iterator:
            img, label = data  # label here is not so important
            VAE_GAN_train_one_epoch(iterator,
                                    img,
                                    gen,
                                    discrim,
                                    criterion,
                                    optim_Dis,
                                    optim_E,
                                    optim_D,
                                    Wassertein,
                                    gp,
                                    gamma,
                                    lamda=lamda)
            if Wassertein:
                for dis in discrim.main:
                    if dis.__class__.__name__ == ('Linear' or 'Conv2d'):
                        dis.weight.requires_grad_ = False
                        dis.weight.data.clamp_(-threshold, threshold)
                        dis.weight.requires_grad_ = True
            scheduler_E.step()
            scheduler_D.step()
            scheduler_Dis.step()
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    torch.save(gen.decoder.state_dict(), "checkpoint/Generate_model.pt")
    return gen.decoder


def train_gan(raw_dataset,
              checkpoint=None,
              epochs=10,
              batch_size=64,
              nz=100,
              device="cuda:0",
              lr=1e-3,
              training=True,
              spectral_norm=True,
              self_attn=False):
    netG, netD, criterion, optimizerG, optimizerD, lr_schedulerG, lr_schedulerD, emaG = load_everything(nz,
                                                                                                        device,
                                                                                                        lr,
                                                                                                        spectral_norm,
                                                                                                        self_attn)
    # Training Loop
    if not training:
        return netG, checkpoint
    # Lists to keep track of progress
    img_list = []
    real_label = 1.
    fake_label = 0.
    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

    for g in range(params.k):
        G_losses = []
        D_losses = []
        iters = 0
        dataloader = load_dataset(raw_dataset, batch_size, g)
        print("Starting Training Loop for group{}...".format(g))
        # For each epoch
        for epoch in range(epochs):
            # For each batch in the dataloader
            for i, data in enumerate(dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # Train with all-real batch
                netD.zero_grad()
                # Format batch
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                Valid_label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                Fake_label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
                # Forward pass real batch through D
                output = netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, Valid_label)
                # Calculate gradients for D in backward pass
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                # Generate fake image batch with G
                fake = netG(noise)
                # Classify all fake batch with D
                output = netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, Fake_label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                errD.backward()
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, Valid_label)
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
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        torch.save(netG.state_dict(), "checkpoint/Generate_model_trained_group{}.pt".format(g))
        torch.cuda.empty_cache()
        plot(G_losses, D_losses)
    return netG, img_list


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
    print("-----------------------------------Start Generating Images---------------------------------------------")
    width, height = 64, 128
    modelG.eval()
    if modelG_checkpoint and os.path.exists(modelG_checkpoint):
        modelG.load_state_dict(torch.load(modelG_checkpoint), strict=False)
    # experimental
    modelG = torch.jit.script(modelG)
    if not os.path.exists("synthetic_images/"):
        os.mkdir("synthetic_images/")
    for i in range(params.instances):
        with torch.no_grad():
            if params.vae:
                fixed_noise = torch.randn(1, 128).to(device)
            else:
                fixed_noise = torch.randn(1, 100, 1, 1).to(device)
            generated_image = modelG(fixed_noise)
        # emaG.restore()

        generated_image = generated_image.squeeze()
        generated_image = transforms.Resize((width, height))(generated_image)
        generated_image = generated_image.cpu().detach().numpy().transpose(1, 2, 0)
        generated_image = Image.fromarray(generated_image, "RGB")
        generated_image.save("synthetic_images/synthetic_image{}.jpg".format(str(i).zfill(5)))
    print("-----------------------------------Generating Images Completed-----------------------------------------")


def parser():
    args = argparse.ArgumentParser()
    args.add_argument("--root", type=str, default="~")
    args.add_argument("--epochs", type=int, default=20)
    args.add_argument("--bs", type=int, default=64)
    args.add_argument("--k", type=int, default=1, help="number of clusters in k-means")
    args.add_argument("--vae", action="store_true")
    args.add_argument("--Wassertein", action="store_true")
    args.add_argument("--gp", action="store_true")
    args.add_argument("--instances", default=1000, type=int)
    args.add_argument("--gamma", default=20, type=int)
    args.add_argument("--self_attn", action="store_true")
    return args.parse_args()


if __name__ == "__main__":
    params = parser()
    query_images = fetch_rawdata("/".join((params.root, "Market1501/bounding_box_train/")),
                                 "/".join((params.root, "Market1501/bounding_box_test/")))
    raw_dataset, num_classes = construct_raw_dataset(query_images)
    torch.cuda.empty_cache()
    if params.k > 1:
        reid_dataset = DataSet4GAN(raw_dataset, params.root, transform=transforms.Compose([
                transforms.Resize((128, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]))
        groups = get_groups(reid_dataset, params.k)
        raw_dataset = list(zip(*raw_dataset, groups))
    if params.vae:
        netG = train_VAE_GAN(raw_dataset,
                             epochs=params.epochs,
                             batch_size=params.bs,
                             gamma=params.gamma,
                             Wassertein=params.Wassertein,
                             gp=params.gp,
                             self_attn=params.self_attn)
    else:
        netG, img_list = train_gan(raw_dataset, epochs=params.epochs, batch_size=params.bs, self_attn=params.self_attn)
        plot_imglist(img_list)
    generate(None, netG)
