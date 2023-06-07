import torch
import torch.nn as nn
from .discrim import SelfAttention


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# try with VAE-GAN
class VAE(nn.Module):
    def __init__(self, spectral_norm=False, self_attn=False, device="cuda"):
        super(VAE, self).__init__()
        if spectral_norm:
            self.encoder = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(3, 64, 5, stride=2, padding=2, bias=False)),
                nn.BatchNorm2d(64, momentum=0.9),
                nn.ReLU(True),
                nn.utils.spectral_norm(nn.Conv2d(64, 128, 5, stride=2, padding=2, bias=False)),
                nn.BatchNorm2d(128, momentum=0.9),
                nn.ReLU(True),
                nn.utils.spectral_norm(nn.Conv2d(128, 256, 5, stride=2, padding=2, bias=False)),
                nn.BatchNorm2d(256, momentum=0.9),
                nn.LeakyReLU(0.2),
                nn.Flatten(),
                nn.Linear(256*16*8, 2048), # Is it necessary?
                nn.BatchNorm1d(2048, momentum=0.9),
                nn.ReLU(True)
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 5, stride=2, padding=2, bias=False),
                nn.BatchNorm2d(64, momentum=0.9),
                nn.ReLU(True),
                nn.Conv2d(64, 128, 5, stride=2, padding=2, bias=False),
                nn.BatchNorm2d(128, momentum=0.9),
                nn.ReLU(True),
                nn.Conv2d(128, 256, 5, stride=2, padding=2, bias=False),
                nn.BatchNorm2d(256, momentum=0.9),
                nn.LeakyReLU(0.2),
                nn.Flatten(),
                nn.Linear(256*16*8, 2048),
                nn.BatchNorm1d(2048, momentum=0.9),
                nn.ReLU(True)
            )
        self.fc_mean = nn.Linear(2048, 128)
        self.fc_var = nn.Linear(2048, 128)

        if spectral_norm:
            self.decoder = nn.Sequential(
                nn.Linear(128, 16 * 8 * 256),
                nn.BatchNorm1d(16 * 8 * 256, momentum=0.9),
                nn.LeakyReLU(0.2),
                nn.Unflatten(1, torch.Size([256, 16, 8])),
                nn.utils.spectral_norm(nn.ConvTranspose2d(256, 256, 6, stride=2, padding=2, bias=False), dim=1),
                nn.BatchNorm2d(256, momentum=0.9),
                nn.utils.spectral_norm(nn.ConvTranspose2d(256, 128, 6, stride=2, padding=2, bias=False), dim=1),
                nn.BatchNorm2d(128, momentum=0.9),
                SelfAttention(128) if self_attn else nn.Identity(),
                nn.utils.spectral_norm(nn.ConvTranspose2d(128, 32, 6, stride=2, padding=2, bias=False), dim=1),
                nn.BatchNorm2d(32, momentum=0.9),
                SelfAttention(32) if self_attn else nn.Identity(),
                nn.utils.spectral_norm(nn.ConvTranspose2d(32, 3, 5, stride=1, padding=2), dim=1),
                nn.Tanh()
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(128, 16*8*256),
                nn.BatchNorm1d(16*8*256, momentum=0.9),
                nn.LeakyReLU(0.2),
                nn.Unflatten(1, torch.Size([256, 16, 8])),
                nn.ConvTranspose2d(256, 256, 6, stride=2, padding=2, bias=False),  # 5 or 6?
                nn.BatchNorm2d(256, momentum=0.9),
                nn.ConvTranspose2d(256, 128, 6, stride=2, padding=2, bias=False),
                nn.BatchNorm2d(128, momentum=0.9),
                SelfAttention(128) if self_attn else nn.Identity(),
                nn.ConvTranspose2d(128, 32, 6, stride=2, padding=2, bias=False),
                nn.BatchNorm2d(32, momentum=0.9),
                SelfAttention(32) if self_attn else nn.Identity(),
                nn.ConvTranspose2d(32, 3, 5, stride=1, padding=2),
                nn.Tanh()
            )
        self.device = device

    def forward(self, x):
        bs = x.size()[0]
        encoded = self.encoder(x)
        mean = self.fc_mean(encoded)
        var = self.fc_var(encoded)
        epsilon = torch.randn(bs, 128).to(self.device)
        z = mean + var * epsilon
        x_tilda = self.decoder(z)
        return mean, var, x_tilda


class Generator(nn.Module):
    def __init__(self, ngpu=1, nz=100, ngf=64, nc=3, spectral_norm=False, self_attn=False):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        if spectral_norm:
            self.main = nn.Sequential(
                nn.utils.spectral_norm(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), dim=1),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                nn.utils.spectral_norm(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), dim=1),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                nn.utils.spectral_norm(nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), dim=1),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                SelfAttention(ngf * 2) if self_attn else nn.Identity(),
                nn.utils.spectral_norm(nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), dim=1),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                SelfAttention(ngf) if self_attn else nn.Identity(),
                nn.ConvTranspose2d(ngf, nc, (6, 4), (4, 2), 1, bias=False), # ?
                nn.Tanh()
            )
        else:
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
                SelfAttention(ngf * 2) if self_attn else nn.Identity(),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                SelfAttention(ngf) if self_attn else nn.Identity(),
                nn.ConvTranspose2d(ngf, nc, (6, 4), (4, 2), 1, bias=False), # ?
                nn.Tanh()
                # state size. (nc) x 128 x 64
            )

    def forward(self, input):
        return self.main(input)
