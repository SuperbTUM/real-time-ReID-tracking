# credit to @mf1024
import torch
import torch.nn as nn


class BatchRenormalization2D(nn.Module):
    """
    This reimplementation is a bit tricky
    Originally the r_max and d_max are quickly converged
    But it should not be according to the paper (~1/4 of training process)
    """

    def __init__(self,
                 num_features,
                 eps=1e-05,
                 momentum=0.01,
                 r_d_max_inc_step=1e-5,
                 r_max=1.0,
                 d_max=0.0):
        super(BatchRenormalization2D, self).__init__()

        self.eps = eps
        self.momentum = torch.tensor(momentum)

        self.gamma = torch.nn.Parameter(torch.ones((1, num_features, 1, 1)), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.zeros((1, num_features, 1, 1)), requires_grad=True)

        self.running_avg_mean = torch.ones((1, num_features, 1, 1), requires_grad=True)
        self.running_avg_std = torch.zeros((1, num_features, 1, 1), requires_grad=True)

        self.max_r_max = 3.0
        self.max_d_max = 5.0

        self.r_max_inc_step = r_d_max_inc_step
        self.d_max_inc_step = r_d_max_inc_step

        self.r_max = torch.tensor(r_max)
        self.d_max = torch.tensor(d_max)

        self.step_counter = 0

    def forward(self, x):
        
        device = x.device
        
        self.running_avg_std = self.running_avg_std.to(device)
        self.running_avg_mean = self.running_avg_mean.to(device)
        
        self.gamma = self.gamma.to(device)
        self.beta = self.beta.to(device)

        self.r_max = self.r_max.to(device)
        self.d_max = self.d_max.to(device)

        batch_ch_mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
        batch_ch_std = torch.clamp(torch.std(x, dim=(0, 2, 3), keepdim=True), self.eps, 1e10)

        if self.training:
            self.step_counter += 1
            r = torch.clamp(batch_ch_std / self.running_avg_std, 1.0 / self.r_max, self.r_max).data
            d = torch.clamp((batch_ch_mean - self.running_avg_mean) / self.running_avg_std, -self.d_max, self.d_max).data

            x = ((x - batch_ch_mean) * r) / batch_ch_std + d
            x = self.gamma * x + self.beta

            if self.step_counter > 5000 and self.r_max < self.max_r_max:
                self.r_max += 0.5 * self.r_max_inc_step * x.shape[0]

            if self.step_counter > 2000 and self.d_max < self.max_d_max:
                self.d_max += 2 * self.d_max_inc_step * x.shape[0]

            self.running_avg_mean = self.running_avg_mean + self.momentum * (
                        batch_ch_mean.data - self.running_avg_mean)
            self.running_avg_std = self.running_avg_std + self.momentum * (
                        batch_ch_std.data - self.running_avg_std)

        else:
            x = (x - self.running_avg_mean) / self.running_avg_std
            x = self.gamma * x + self.beta

        return x


class BatchRenormalization2D_Noniid(BatchRenormalization2D):
    """Dedicated for metric learning where sampling is non-iid"""
    def __init__(self, num_features, num_instance, eps=1e-05, momentum=0.01, r_d_max_inc_step=1e-5):
        super(BatchRenormalization2D_Noniid, self).__init__(num_features, eps, momentum, r_d_max_inc_step)
        self.num_instance = num_instance

    def forward(self, x):
        x_splits = []
        for i in range(self.num_instance):
            x_split = []
            for j in range(x.size(0) // self.num_instance):
                x_split.append(x[i+self.num_instance*j])
            x_splits.append(torch.stack(x_split))
        x_normed = [torch.tensor(0.) for _ in range(x.size(0))]

        device = x.device

        self.running_avg_std = self.running_avg_std.to(device)
        self.running_avg_mean = self.running_avg_mean.to(device)

        self.gamma = self.gamma.to(device)
        self.beta = self.beta.to(device)

        self.r_max = self.r_max.to(device)
        self.d_max = self.d_max.to(device)

        for i, x_mini in enumerate(x_splits):

            batch_ch_mean = torch.mean(x_mini, dim=(0, 2, 3), keepdim=True)
            batch_ch_std = torch.clamp(torch.std(x_mini, dim=(0, 2, 3), keepdim=True), self.eps, 1e10)

            if self.training:
                r = torch.clamp(batch_ch_std / self.running_avg_std, 1.0 / self.r_max, self.r_max).data
                d = torch.clamp((batch_ch_mean - self.running_avg_mean) / self.running_avg_std, -self.d_max,
                                self.d_max).data

                x_mini = ((x_mini - batch_ch_mean) * r) / batch_ch_std + d
                x_mini = self.gamma * x_mini + self.beta

                self.running_avg_mean = self.running_avg_mean + self.momentum * (
                        batch_ch_mean.data - self.running_avg_mean)
                self.running_avg_std = self.running_avg_std + self.momentum * (
                        batch_ch_std.data - self.running_avg_std)

            else:
                x_mini = (x_mini - self.running_avg_mean) / self.running_avg_std
                x_mini = self.gamma * x_mini + self.beta

            for j in range(x_mini.size(0)):
                x_normed[self.num_instance * j + i % self.num_instance] = x_mini[j]

        self.step_counter += 1
        if self.step_counter > 5000 and self.r_max < self.max_r_max:
            self.r_max += 0.5 * self.r_max_inc_step * x.shape[0]

        if self.step_counter > 2000 and self.d_max < self.max_d_max:
            self.d_max += 2 * self.d_max_inc_step * x.shape[0]

        x_normed = torch.stack(x_normed, dim=0)
        return x_normed


class BatchRenormalization1D(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.01, r_d_max_inc_step=1e-5):
        super(BatchRenormalization1D, self).__init__()

        self.eps = eps
        self.momentum = torch.tensor(momentum)

        self.gamma = torch.nn.Parameter(torch.ones((1, num_features)), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.zeros((1, num_features)), requires_grad=True)

        self.running_avg_mean = torch.ones((1, num_features), requires_grad=False)
        self.running_avg_std = torch.zeros((1, num_features), requires_grad=False)

        self.max_r_max = 3.0
        self.max_d_max = 5.0

        self.r_max_inc_step = r_d_max_inc_step
        self.d_max_inc_step = r_d_max_inc_step

        self.r_max = torch.tensor(1.0)
        self.d_max = torch.tensor(0.0)

    def forward(self, x):

        device = x.device

        self.running_avg_std = self.running_avg_std.to(device)
        self.running_avg_mean = self.running_avg_mean.to(device)

        self.gamma = self.gamma.to(device)
        self.beta = self.beta.to(device)

        self.r_max = self.r_max.to(device)
        self.d_max = self.d_max.to(device)

        batch_ch_mean = torch.mean(x, dim=0, keepdim=True)
        batch_ch_std = torch.clamp(torch.std(x, dim=0, keepdim=True), self.eps, 1e10)

        if self.training:
            r = torch.clamp(batch_ch_std / self.running_avg_std, 1.0 / self.r_max, self.r_max).data
            d = torch.clamp((batch_ch_mean - self.running_avg_mean) / self.running_avg_std, -self.d_max,
                            self.d_max).data

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
