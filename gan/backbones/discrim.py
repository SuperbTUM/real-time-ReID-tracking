import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, 3, 1, 1))
        self.downsample_layer = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1, 1, 0))

        nn.init.xavier_uniform_(self.conv1.weight.data, math.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, math.sqrt(2))
        nn.init.xavier_uniform_(self.downsample_layer.weight.data)

    def forward(self, x):
        branch = x
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(self.conv2(x), 2)
        branch = F.avg_pool2d(branch, 2)
        branch = self.downsample_layer(branch)
        x += branch
        return x


class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class Discriminator(nn.Module):
    def __init__(self,
                 ngpu=1,
                 nc=3,
                 ndf=64,
                 VAE=False,
                 Wassertein=False,
                 spectral_norm=False,
                 self_attn=False):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        if Wassertein:
            self.main = nn.Sequential(
                # input is (nc) x 128 x 64
                nn.Conv2d(nc, ndf, 4, (4, 2), 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # # state size. (ndf) x  x 32
                # nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                # nn.BatchNorm2d(ndf * 2),
                # nn.LeakyReLU(0.2, inplace=True),
                # # state size. (ndf*2) x 16 x 16
                # nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                # nn.BatchNorm2d(ndf * 4),
                # nn.LeakyReLU(0.2, inplace=True),
                # # state size. (ndf*4) x 8 x 8
                # nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                # nn.BatchNorm2d(ndf * 8),
                # nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Conv2d(ndf, ndf * 2, 3, stride=1, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d((2, 2)),

                nn.Conv2d(ndf * 2, ndf * 4, 3, stride=1, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d((2, 2)),

                nn.Conv2d(ndf * 4, ndf * 8, 3, stride=1, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d((2, 2)),
            )
        else:
            if spectral_norm:
                self.main = nn.Sequential(
                    BasicBlock(nc, ndf),
                    BasicBlock(ndf, ndf * 2),
                    BasicBlock(ndf * 2, ndf * 4),
                    BasicBlock(ndf * 4, ndf * 8),
                    nn.ReLU(True),
                )
            else:
                self.main = nn.Sequential(
                    # input is (nc) x 128 x 64
                    nn.Conv2d(nc, ndf, 4, (4, 2), 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf) x  x 32
                    nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf*2) x 16 x 16
                    nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf*4) x 8 x 8
                    SelfAttention(ndf * 4) if self_attn else nn.Identity(),
                    nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf*8) x 4 x 4
                    # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                )
        self.extension = nn.Sequential(
            nn.Linear(ndf*8, 512),
            nn.BatchNorm1d(512, momentum=0.9),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 1)
        )
        self.attn = SelfAttention(ndf * 8)
        self.getDis = nn.Linear(ndf * 8, 1, bias=False)#nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.VAE = VAE
        self.Wassertein = Wassertein
        self.self_attn = self_attn

    def forward(self, input):
        bs = input.size(0)
        main = self.main(input)
        if self.self_attn:
            main = self.attn(main)
        main = F.adaptive_avg_pool2d(main, 1)
        main = main.view(bs, -1)
        if self.VAE:
            main1 = main
            if self.Wassertein:
                return self.extension(main), main1
            return self.sigmoid(self.extension(main)), main1
        if self.Wassertein:
            return self.getDis(main)
        get_dis = self.getDis(main)
        return self.sigmoid(get_dis)
