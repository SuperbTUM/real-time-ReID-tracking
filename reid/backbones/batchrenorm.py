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
                 dict_state=None,
                 eps=1e-05,
                 momentum=0.1,
                 r_d_max_inc_step=1e-5,
                 r_max=1.0,
                 d_max=0.0):
        super(BatchRenormalization2D, self).__init__()

        self.eps = eps
        self.momentum = torch.tensor(momentum)

        self.gamma = nn.Parameter(torch.ones((1, num_features, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, num_features, 1, 1)), requires_grad=True)

        self.register_buffer('running_avg_mean', torch.zeros((1, num_features, 1, 1)))
        self.register_buffer('running_avg_std', torch.ones((1, num_features, 1, 1)))
        self.register_buffer('num_tracked_batch', torch.tensor(0)) # in case momentum is None

        self.max_r_max = 3.0
        self.max_d_max = 5.0

        self.r_max_inc_step = r_d_max_inc_step
        self.d_max_inc_step = r_d_max_inc_step

        self.register_buffer('r_max', torch.tensor(r_max))
        self.register_buffer('d_max', torch.tensor(d_max))

        self.dict_state = dict_state
        self._load_params_from_bn()

    def _load_params_from_bn(self):
        if self.dict_state is None:
            return
        weight = self.dict_state['weight'].data
        weight = weight.reshape(1, weight.size(0), 1, 1)
        bias = self.dict_state['bias'].data
        bias = bias.reshape(1, bias.size(0), 1, 1)
        running_mean = self.dict_state['running_mean'].data
        running_mean = running_mean.reshape(1, running_mean.size(0), 1, 1)
        running_var = self.dict_state['running_var']
        running_var = running_var.data.reshape(1, running_var.size(0), 1, 1)

        self.gamma.data = weight.clone()
        self.beta.data = bias.clone()
        self.running_avg_mean.data = running_mean.clone()
        self.running_avg_std.data = running_var.clone()

    def forward(self, x):

        batch_ch_mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
        # in version 2.0: correction, otherwise: unbiased=False
        batch_ch_std = torch.clamp(torch.std(x, dim=(0, 2, 3), correction=0, keepdim=True), self.eps, 1e10)

        if self.training:
            self.num_tracked_batch += 1
            r = torch.clamp(batch_ch_std / self.running_avg_std, 1.0 / self.r_max, self.r_max).data
            d = torch.clamp((batch_ch_mean - self.running_avg_mean) / self.running_avg_std, -self.d_max, self.d_max).data

            x = ((x - batch_ch_mean) * r) / batch_ch_std + d
            x = self.gamma * x + self.beta

            if self.num_tracked_batch > 5000 and self.r_max < self.max_r_max:
                self.r_max += 0.5 * self.r_max_inc_step * x.shape[0]

            if self.num_tracked_batch > 2000 and self.d_max < self.max_d_max:
                self.d_max += 2 * self.d_max_inc_step * x.shape[0]

            self.running_avg_mean = self.running_avg_mean + self.momentum * (batch_ch_mean.data - self.running_avg_mean)
            self.running_avg_std = self.running_avg_std + self.momentum * (batch_ch_std.data - self.running_avg_std)

        else:
            x = (x - self.running_avg_mean) / self.running_avg_std
            x = self.gamma * x + self.beta

        return x


class BatchRenormalization2D_Noniid(BatchRenormalization2D):
    """Dedicated for metric learning where sampling is non-iid"""
    def __init__(self, num_features, num_instance, dict_state=None, eps=1e-05, momentum=0.1, r_d_max_inc_step=1e-5):
        super(BatchRenormalization2D_Noniid, self).__init__(num_features, dict_state, eps, momentum, r_d_max_inc_step)
        self.num_instance = num_instance

    def forward(self, x):
        x_splits = []
        for i in range(self.num_instance):
            x_split = []
            for j in range(x.size(0) // self.num_instance):
                x_split.append(x[i+self.num_instance*j])
            x_splits.append(torch.stack(x_split))
        x_normed = [torch.tensor(0.) for _ in range(x.size(0))]

        for i, x_mini in enumerate(x_splits):

            batch_ch_mean = torch.mean(x_mini, dim=(0, 2, 3), keepdim=True)
            # in version 2.0: correction, otherwise: unbiased=False
            batch_ch_std = torch.clamp(torch.std(x_mini, dim=(0, 2, 3), correction=0, keepdim=True), self.eps, 1e10)

            if self.training:
                r = torch.clamp(batch_ch_std / self.running_avg_std, 1.0 / self.r_max, self.r_max).data
                d = torch.clamp((batch_ch_mean - self.running_avg_mean) / self.running_avg_std, -self.d_max,
                                self.d_max).data

                x_mini = ((x_mini - batch_ch_mean) * r) / batch_ch_std + d
                x_mini = self.gamma * x_mini + self.beta

                self.running_avg_mean = self.running_avg_mean + self.momentum * (batch_ch_mean.data - self.running_avg_mean)
                self.running_avg_std = self.running_avg_std + self.momentum * (batch_ch_std.data - self.running_avg_std)

            else:
                x_mini = (x_mini - self.running_avg_mean) / self.running_avg_std
                x_mini = self.gamma * x_mini + self.beta

            for j in range(x_mini.size(0)):
                x_normed[self.num_instance * j + i % self.num_instance] = x_mini[j]

        self.num_tracked_batch += 1
        if self.num_tracked_batch > 5000 and self.r_max < self.max_r_max:
            self.r_max += 0.5 * self.r_max_inc_step * x.shape[0]

        if self.num_tracked_batch > 2000 and self.d_max < self.max_d_max:
            self.d_max += 2 * self.d_max_inc_step * x.shape[0]

        x_normed = torch.stack(x_normed, dim=0)
        return x_normed


class BatchRenormalization1D(nn.Module):

    def __init__(self,
                 num_features,
                 dict_state=None,
                 eps=1e-05,
                 momentum=0.1,
                 r_d_max_inc_step=1e-5,
                 r_max=1.0,
                 d_max=0.0):
        super(BatchRenormalization1D, self).__init__()

        self.eps = eps
        self.momentum = torch.tensor(momentum)

        self.gamma = nn.Parameter(torch.ones((1, num_features)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, num_features)), requires_grad=True)

        self.register_buffer('running_avg_mean', torch.zeros((1, num_features)))
        self.register_buffer('running_avg_std', torch.ones((1, num_features)))
        self.register_buffer('num_tracked_batch', torch.tensor(0))  # in case momentum is None

        self.max_r_max = 3.0
        self.max_d_max = 5.0

        self.r_max_inc_step = r_d_max_inc_step
        self.d_max_inc_step = r_d_max_inc_step

        self.register_buffer('r_max', torch.tensor(r_max))
        self.register_buffer('d_max', torch.tensor(d_max))

        self.dict_state = dict_state
        self._load_params_from_bn()

    def _load_params_from_bn(self):
        if self.dict_state is None:
            return
        weight = self.dict_state['weight'].data
        weight = weight.reshape(1, weight.size(0))
        bias = self.dict_state['bias'].data
        bias = bias.reshape(1, bias.size(0))
        running_mean = self.dict_state['running_mean'].data
        running_mean = running_mean.reshape(1, running_mean.size(0))
        running_var = self.dict_state['running_var']
        running_var = running_var.data.reshape(1, running_var.size(0))

        self.gamma.data = weight.clone()
        self.beta.data = bias.clone()
        self.running_avg_mean.data = running_mean.clone()
        self.running_avg_std.data = running_var.clone()

    def forward(self, x):

        batch_ch_mean = torch.mean(x, dim=0, keepdim=True)
        # in version 2.0: correction, otherwise: unbiased=False
        batch_ch_std = torch.clamp(torch.std(x, dim=0, correction=0, keepdim=True), self.eps, 1e10)

        if self.training:
            r = torch.clamp(batch_ch_std / self.running_avg_std, 1.0 / self.r_max, self.r_max).data
            d = torch.clamp((batch_ch_mean - self.running_avg_mean) / self.running_avg_std, -self.d_max,
                            self.d_max).data

            x = ((x - batch_ch_mean) * r) / batch_ch_std + d
            x = self.gamma * x + self.beta

            if self.r_max < self.max_r_max:
                self.r_max += 0.5 * self.r_max_inc_step * x.shape[0]

            if self.d_max < self.max_d_max:
                self.d_max += 2 * self.d_max_inc_step * x.shape[0]

            self.running_avg_mean = self.running_avg_mean + self.momentum * (batch_ch_mean.data - self.running_avg_mean)
            self.running_avg_std = self.running_avg_std + self.momentum * (batch_ch_std.data - self.running_avg_std)

        else:
            x = (x - self.running_avg_mean) / self.running_avg_std
            x = self.gamma * x + self.beta

        return x
