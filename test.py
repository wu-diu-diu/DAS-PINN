import torch
import numpy as np


class DiagGaussian():
    def __init__(self, mu, cov):
        self.shape = mu.shape
        self.d = np.prod(self.shape)  ## prod计算数组所有元素的乘积。 这里计算mu的总维度
        self.loc = mu  ## 均值
        self.scale = torch.sqrt(torch.diagonal(cov)).view(-1,)  ## 计算cov的对角线元素的平方根即标准差 展开成一维tensor
        self.log_scale = torch.log(self.scale)
    def forward(self, num_samples=1):
        eps = torch.randn((num_samples,) + self.shape, dtype=self.loc.dtype, device=self.loc.device) ## (num_samples,) + (2,4) = (num_samples, 2, 4)
        log_scale = self.log_scale

        z = self.loc + self.scale * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow(eps, 2), -1
        )
        return z, log_p

    def sample(self, sample_shape=torch.Size()):
        z, log_p = self.forward(sample_shape[0])
        return z
    def log_prob(self, z):
        log_scale = self.log_scale
        # qurad_form = torch.pow((z - self.loc) / torch.exp(log_scale), 2)
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow((z - self.loc) / self.scale, 2),
           dim=-1)
        return log_p


x = torch.randn(10,3)
mu = torch.zeros((1,3))
cov = torch.eye(3)
test = DiagGaussian(mu, cov)
z, log = test.forward(10)