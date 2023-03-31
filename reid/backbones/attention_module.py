###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import torch
from torch.nn import Module, Conv2d, Parameter, Softmax
import torch.nn as nn

import logging

class SEModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        #x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class PAM_Module(Module):
    """ Position attention module"""
    # Ref from SAGAN

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.channel_in = in_dim

        self.query_conv = Conv2d(
            in_channels=in_dim,
            out_channels=in_dim // 8,
            kernel_size=1
        )
        self.key_conv = Conv2d(
            in_channels=in_dim,
            out_channels=in_dim // 8,
            kernel_size=1
        )
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(in_dim)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self\
            .query_conv(x)\
            .view(m_batchsize, -1, width * height)\
            .permute(0, 2, 1)
        proj_key = self\
            .key_conv(x)\
            .view(m_batchsize, -1, width * height)
        #proj_query = proj_query * (1 - 1 / (C/8))
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        x_view = x.view(m_batchsize, -1, width * height)

        out = torch.bmm(x_view, attention.permute(0, 2, 1))
        attention_mask = out.view(m_batchsize, C, height, width)

        out = self.gamma * attention_mask
        out = self.bn(out)
        out = out + x
        return out

class Attention_Module(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.in_channel = in_dim
        self.pam = PAM_Module(in_dim)
        self.se = SEModule(in_dim)

    def forward(self, x):
        out = self.pam(x)
        out = self.se(out)
        return out
