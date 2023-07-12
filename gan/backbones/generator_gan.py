import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .discriminator_gan import SelfAttention
from .categorical_conditional_bn import CategoricalConditionalBatchNorm2d


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_class=0, upsample=True, ratio=(2, 2)):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        if num_class:
            self.bn1 = CategoricalConditionalBatchNorm2d(num_class, in_channel)
            self.bn2 = CategoricalConditionalBatchNorm2d(num_class, out_channel)
        else:
            self.bn1 = nn.BatchNorm2d(in_channel)
            self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample_layer = nn.Conv2d(in_channel, out_channel, 1, 1, 0)

        nn.init.xavier_uniform_(self.conv1.weight.data, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.downsample_layer.weight.data)
        self.upsample = upsample
        self.ratio = ratio

    def _upsample(self, x):
        h, w = x.size()[2:]
        return F.interpolate(x, size=(h*self.ratio[0], w*self.ratio[1]), mode="bilinear")

    def forward(self, x, y=None, **kwargs):
        branch = x
        if y is not None:
            x = self.bn1(x, y, **kwargs)
        else:
            x = self.bn1(x)
        x = F.relu(x)
        if self.upsample:
            x = self._upsample(x)
            branch = self._upsample(branch)
        x = self.conv1(x)
        if y is not None:
            x = self.bn2(x, y, **kwargs)
        else:
            x = self.bn2(x)
        x = self.conv2(F.relu(x))
        branch = self.downsample_layer(branch)
        x += branch
        return x


# try with VAE-GAN
class VAE(nn.Module):
    def __init__(self, spectral_norm=False, self_attn=False, device="cuda"):
        super(VAE, self).__init__()
        if spectral_norm:
            self.encoder = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(3, 64, 5, stride=2, padding=2, bias=False)),
                nn.ReLU(True),
                nn.utils.spectral_norm(nn.Conv2d(64, 128, 5, stride=2, padding=2, bias=False)),
                nn.ReLU(True),
                nn.utils.spectral_norm(nn.Conv2d(128, 256, 5, stride=2, padding=2, bias=False)),
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
                nn.utils.spectral_norm(nn.ConvTranspose2d(256, 128, 6, stride=2, padding=2, bias=False), dim=1),
                SelfAttention(128) if self_attn else nn.Identity(),
                nn.utils.spectral_norm(nn.ConvTranspose2d(128, 32, 6, stride=2, padding=2, bias=False), dim=1),
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
    def __init__(self, nz=100, ngf=64, nc=3, spectral_norm=False, self_attn=False, num_class=0, bottom_width=4):
        super(Generator, self).__init__()
        if spectral_norm:
            self.main = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nz, (bottom_width ** 2 >> 1) * ngf),
                nn.Unflatten(1, (ngf, bottom_width, bottom_width >> 1)),
                BasicBlock(ngf, ngf, num_class=num_class),
                BasicBlock(ngf, ngf, num_class=num_class),
                BasicBlock(ngf, ngf, num_class=num_class),
                SelfAttention(ngf) if self_attn else nn.Identity(),
                BasicBlock(ngf, ngf, num_class=num_class),
                SelfAttention(ngf) if self_attn else nn.Identity(),
                BasicBlock(ngf, ngf, num_class=num_class),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                nn.Conv2d(ngf, nc, 3, 1, 1),
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
