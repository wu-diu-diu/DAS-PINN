import torch
import numpy as np
import torch.nn as nn


class BatchNorm(nn.Module):
    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        """
        input_size = features
        """
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.log_gamma = nn.Parameter(torch.ones(input_size))  ## add log for logdet
        self.beta = nn.Parameter(torch.zeros(input_size))
        ## 将runningmean保存在buffer中 不会被更新
        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))

    def forward(self, x):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0)
            ## updating running_mean = runing_mean * momentum + batch_mena * (1 - momentum)
            self.running_mean.mul_(self.momentum).add_(self.batch_mean * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var * (1 - self.momentum))
            mean = self.batch_mean
            var = self.batch_var
        else:  ## inference
            mean = self.running_mean
            var = self.running_var
        ## computing
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
        return y, log_abs_det_jacobian.expand_as(x)  ## expand_as是将1*2 复制每一行扩展成10*2即只能通过复制原有数据进行expand

    def inverse(self, y):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean
        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma
        return x, log_abs_det_jacobian.expand_as(x)
