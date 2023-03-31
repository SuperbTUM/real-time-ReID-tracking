# credit to @mf1024
import torch
import torch.nn as nn


class BatchRenormalization2D(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.01, r_d_max_inc_step=0.0001):
        super(BatchRenormalization2D, self).__init__()

        self.eps = eps
        self.momentum = torch.tensor(momentum)

        self.gamma = torch.nn.Parameter(torch.ones((1, num_features, 1, 1)), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.zeros((1, num_features, 1, 1)), requires_grad=True)

        self.running_avg_mean = torch.ones((1, num_features, 1, 1), requires_grad=False)
        self.running_avg_std = torch.zeros((1, num_features, 1, 1), requires_grad=False)

        self.max_r_max = 3.0
        self.max_d_max = 5.0

        self.r_max_inc_step = r_d_max_inc_step
        self.d_max_inc_step = r_d_max_inc_step

        self.r_max = torch.tensor(1.0)
        self.d_max = torch.tensor(0.0)

    def forward(self, x):

        batch_ch_mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
        batch_ch_std = torch.clamp(torch.std(x, dim=(0, 2, 3), keepdim=True), self.eps, 1e10)

        if self.training:
            r = torch.clamp(batch_ch_std / self.running_avg_std, 1.0 / self.r_max, self.r_max).data
            d = torch.clamp((batch_ch_mean - self.running_avg_mean) / self.running_avg_std, -self.d_max, self.d_max).data

            x = ((x - batch_ch_mean) * r) / batch_ch_std + d
            x = self.gamma * x + self.beta

            if self.r_max < self.max_r_max:
                self.r_max += self.r_max_inc_step * x.shape[0]

            if self.d_max < self.max_d_max:
                self.d_max += self.d_max_inc_step * x.shape[0]

            self.running_avg_mean = self.running_avg_mean + self.momentum * (
                        batch_ch_mean.data - self.running_avg_mean)
            self.running_avg_std = self.running_avg_std + self.momentum * (
                        batch_ch_std.data - self.running_avg_std)

        else:
            x = (x - self.running_avg_mean) / self.running_avg_std
            x = self.gamma * x + self.beta

        return x
