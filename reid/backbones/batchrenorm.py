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
        self.register_buffer('running_avg_var', torch.ones((1, num_features, 1, 1)))
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
        running_var = self.dict_state['running_var'].data
        running_var = running_var.reshape(1, running_var.size(0), 1, 1)

        self.gamma.data = weight.clone()
        self.beta.data = bias.clone()
        self.running_avg_mean.data = running_mean.clone()
        self.running_avg_var.data = running_var.clone()

    def forward(self, x):

        if self.training:
            batch_ch_mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
            # in version 2.0: correction, otherwise: unbiased=False
            batch_ch_var_unbiased = torch.var(x, dim=(0, 2, 3), unbiased=True, keepdim=True)
            batch_ch_var_biased = torch.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True)

            self.num_tracked_batch += 1
            r = torch.clamp(torch.sqrt((batch_ch_var_biased + self.eps) / (self.running_avg_var + self.eps)), 1.0 / self.r_max, self.r_max).data
            d = torch.clamp((batch_ch_mean - self.running_avg_mean) / torch.sqrt(self.running_avg_var + self.eps), -self.d_max, self.d_max).data

            x = ((x - batch_ch_mean) * r) / torch.sqrt(batch_ch_var_biased + self.eps) + d
            x = self.gamma * x + self.beta

            if self.num_tracked_batch > 500 and self.r_max < self.max_r_max:
                self.r_max += 1.2 * self.r_max_inc_step * x.shape[0]

            if self.num_tracked_batch > 500 and self.d_max < self.max_d_max:
                self.d_max += 4.8 * self.d_max_inc_step * x.shape[0]

            self.running_avg_mean = self.running_avg_mean + self.momentum * (batch_ch_mean.data - self.running_avg_mean)
            self.running_avg_var = self.running_avg_var + self.momentum * (batch_ch_var_unbiased.data - self.running_avg_var)

        else:
            x = (x - self.running_avg_mean) / torch.sqrt(self.running_avg_var + self.eps)
            x = self.gamma * x + self.beta

        return x


class BatchRenormalization2D_Noniid(BatchRenormalization2D):
    """Dedicated for metric learning where sampling is non-iid"""
    def __init__(self,
                 num_features,
                 num_instance,
                 dict_state=None,
                 eps=1e-05,
                 momentum=0.1,
                 r_d_max_inc_step=1e-5,
                 r_max=1.0,
                 d_max=0.0):
        self.num_instance = num_instance
        self.num_features = num_features
        super(BatchRenormalization2D_Noniid, self).__init__(num_features, dict_state, eps, momentum, r_d_max_inc_step, r_max, d_max)
        self.inference_momentum = 0.0 # 0.1

    # def forward_train(self, x):
    #     if not self.training:
    #         self.num_instance = 1 # Let it be iid in inference mode
    #     x_splits = []
    #     for i in range(self.num_instance):
    #         x_split = []
    #         for j in range(x.size(0) // self.num_instance):
    #             x_split.append(x[i+self.num_instance*j])
    #         x_splits.append(torch.stack(x_split))
    #     x_normed = torch.zeros_like(x)
    #
    #     for i, x_mini in enumerate(x_splits):
    #
    #         batch_ch_mean = torch.mean(x_mini, dim=(0, 2, 3), keepdim=True)
    #         # in version 2.0: correction, otherwise: unbiased=False
    #         batch_ch_var_unbiased = torch.var(x_mini, dim=(0, 2, 3), unbiased=True, keepdim=True)
    #         batch_ch_var_biased = torch.var(x_mini, dim=(0, 2, 3), unbiased=False, keepdim=True)
    #
    #         r = torch.clamp(torch.sqrt((batch_ch_var_biased + self.eps) / (self.running_avg_var + self.eps)), 1.0 / self.r_max, self.r_max).data
    #         d = torch.clamp((batch_ch_mean - self.running_avg_mean) / torch.sqrt(self.running_avg_var + self.eps), -self.d_max,
    #                         self.d_max).data
    #
    #         x_mini = ((x_mini - batch_ch_mean) * r) / torch.sqrt(batch_ch_var_biased + self.eps) + d
    #         x_mini = self.gamma * x_mini + self.beta
    #
    #         self.running_avg_mean = self.running_avg_mean + self.momentum * (batch_ch_mean.data - self.running_avg_mean)
    #         self.running_avg_var = self.running_avg_var + self.momentum * (batch_ch_var_unbiased.data - self.running_avg_var)
    #
    #         for j in range(x_mini.size(0)):
    #             x_normed[self.num_instance * j + i % self.num_instance] = x_mini[j]
    #
    #     if self.num_tracked_batch > 500 and self.r_max < self.max_r_max:
    #         self.r_max += 1.2 * self.r_max_inc_step * x.shape[0]
    #
    #     if self.num_tracked_batch > 500 and self.d_max < self.max_d_max:
    #         self.d_max += 4.8 * self.d_max_inc_step * x.shape[0]
    #
    #     return x_normed

    def forward_train(self, x):
        """Looks like group normalization"""
        batch_size = x.size(0)
        minibatch_size = batch_size // self.num_instance
        x_splits = []
        for i in range(self.num_instance):
            x_split = x[i:len(x):self.num_instance]
            x_splits.append(x_split)
        x_splits = torch.cat(x_splits, dim=0)

        x_normed = torch.zeros_like(x)

        batch_ch_mean = torch.mean(x_splits, dim=(2, 3), keepdim=True)
        batch_ch_var_pre = torch.mean(x_splits ** 2, dim=(2, 3), keepdim=True)

        x_splits = x_splits.reshape(self.num_instance, minibatch_size, self.num_features, x.size(2), x.size(3))

        batch_ch_mean = batch_ch_mean.view(self.num_instance, minibatch_size, self.num_features, 1, 1)
        batch_ch_var_pre = batch_ch_var_pre.view(self.num_instance, minibatch_size, self.num_features, 1, 1)

        group_ch_mean = batch_ch_mean.mean(dim=1, keepdim=True)
        group_ch_var_pre = batch_ch_var_pre.mean(dim=1, keepdim=True)
        group_ch_var_biased = group_ch_var_pre - group_ch_mean ** 2

        r = torch.clamp(torch.sqrt((group_ch_var_biased + self.eps) / (self.running_avg_var + self.eps)),
                        1.0 / self.r_max, self.r_max).data
        d = torch.clamp((group_ch_mean - self.running_avg_mean) / torch.sqrt(self.running_avg_var + self.eps),
                        -self.d_max,
                        self.d_max).data

        x_splits = ((x_splits - group_ch_mean) * r) / torch.sqrt(group_ch_var_biased + self.eps) + d
        x_splits = self.gamma * x_splits + self.beta

        self.num_tracked_batch += 1
        if self.num_tracked_batch > 500 and self.r_max < self.max_r_max:
            self.r_max += 1.2 * self.r_max_inc_step * x.shape[0]

        if self.num_tracked_batch > 500 and self.d_max < self.max_d_max:
            self.d_max += 4.8 * self.d_max_inc_step * x.shape[0]

        x_splits = x_splits.view(-1, x_splits.size(2), x_splits.size(3), x_splits.size(4))

        indices = torch.arange(0, batch_size)
        x_normed[self.num_instance * (indices % minibatch_size) + indices // minibatch_size] = x_splits[:]

        batch_ch_mean = batch_ch_mean.view(batch_size, self.num_features, 1, 1)
        batch_ch_var_pre = batch_ch_var_pre.view(batch_size, self.num_features, 1, 1)
        batch_ch_var_biased = (batch_ch_var_pre - batch_ch_mean ** 2)

        self.running_avg_mean = self.running_avg_mean + self.momentum * (
                    batch_ch_mean.mean(dim=0, keepdim=True).data - self.running_avg_mean)
        self.running_avg_var = self.running_avg_var + self.momentum * (
                    batch_ch_var_biased.mean(dim=0, keepdim=True).data - self.running_avg_var)

        return x_normed

    # def forward_eval(self, x):
    #     x = (x - self.running_avg_mean) / torch.sqrt(self.running_avg_var + self.eps)
    #     x = self.gamma * x + self.beta
    #     return x

    def forward_eval(self, x):
        batch_ch_mean = torch.mean(x, dim=(2, 3), keepdim=True)
        batch_ch_var_pre = torch.mean(x ** 2, dim=(2, 3), keepdim=True)
        batch_ch_var_biased = (batch_ch_var_pre - batch_ch_mean ** 2)

        running_avg_mean = (1 - self.inference_momentum) * self.running_avg_mean + self.inference_momentum * batch_ch_mean
        running_avg_var = (1 - self.inference_momentum) * self.running_avg_var + self.inference_momentum * batch_ch_var_biased
        x = (x - running_avg_mean) / torch.sqrt(running_avg_var + self.eps)
        x = self.gamma * x + self.beta
        return x

    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_eval(x)


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
        self.register_buffer('running_avg_var', torch.ones((1, num_features)))
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
        self.running_avg_var.data = running_var.clone()

    def forward(self, x):

        batch_ch_mean = torch.mean(x, dim=0, keepdim=True)
        # in version 2.0: correction, otherwise: unbiased=False
        batch_ch_var_unbiased = torch.var(x, dim=0, unbiased=True, keepdim=True)
        batch_ch_var_biased = torch.var(x, dim=0, unbiased=False, keepdim=True)

        if self.training:
            self.num_tracked_batch += 1

            r = torch.clamp(torch.sqrt((batch_ch_var_biased + self.eps) / (self.running_avg_var + self.eps)), 1.0 / self.r_max, self.r_max).data
            d = torch.clamp((batch_ch_mean - self.running_avg_mean) / torch.sqrt(self.running_avg_var + self.eps), -self.d_max,
                            self.d_max).data

            x = ((x - batch_ch_mean) * r) / torch.sqrt(batch_ch_var_biased + self.eps) + d
            x = self.gamma * x + self.beta

            if self.num_tracked_batch > 500 and self.r_max < self.max_r_max:
                self.r_max += 1.2 * self.r_max_inc_step * x.shape[0]

            if self.num_tracked_batch > 500 and self.d_max < self.max_d_max:
                self.d_max += 4.8 * self.d_max_inc_step * x.shape[0]

            self.running_avg_mean = self.running_avg_mean + self.momentum * (batch_ch_mean.data - self.running_avg_mean)
            self.running_avg_var = self.running_avg_var + self.momentum * (batch_ch_var_unbiased.data - self.running_avg_var)

        else:
            x = (x - self.running_avg_mean) / torch.sqrt(self.running_avg_var + self.eps)
            x = self.gamma * x + self.beta

        return x
