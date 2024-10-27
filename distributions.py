import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class DiagGaussian:
    def __init__(self, mu, cov):
        self.shape = mu.shape
        self.d = np.prod(self.shape)  ## prod计算数组所有元素的乘积。 这里计算mu的总维度
        self.loc = mu  ## 均值
        self.cov = cov
        self.scale = torch.sqrt(torch.diagonal(cov)).view(-1, )  ## 计算cov的对角线元素的平方根即标准差 展开成一维tensor
        self.log_scale = torch.log(self.scale)

    def forward(self, num_samples=1):
        ## 生成num 个符合标准高斯分布的多维随机数
        eps = torch.randn((num_samples,) + self.shape, dtype=self.loc.dtype, device=self.loc.device)
        ## (num_samples,) + (2,4) = (num_samples, 2, 4)
        log_scale = self.log_scale
        ## 使得改随机数符合 给定的均值和方差
        z = self.loc + self.scale * eps  ## [num_sample,] + self.shape
        return z

    def sample(self, n_samples):
        z = self.forward(n_samples[0])
        return z

    def log_prob(self, z):
        log_scale = self.log_scale
        # qurad_form = torch.pow((z - self.loc) / torch.exp(log_scale), 2)
        # log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
        #     log_scale + 0.5 * torch.pow((z - self.loc) / self.scale, 2),
        #     dim=-1)
        # tmp = z - self.loc
        # log_p = (-0.5 * self.d * np.log(2 * np.pi) -
        #          0.5 * torch.sum(torch.matmul(torch.matmul(tmp, self.cov), tmp.T), dim=-1))
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow((z - self.loc) / self.scale, 2),
            dim=-1)
        return log_p


def original_dist(num_samples):
    # points, _ = make_circles(n_samples=num_samples, noise=0.06, factor=0.5)
    z = torch.randn(num_samples, 2)
    scale = 4
    sq2 = 1 / np.sqrt(2)
    centers = [(1, 0), (-1, 0), (0, 1), (0, -1),
               (sq2, sq2), (-sq2, sq2), (sq2, -sq2), (-sq2, -sq2)]
    centers = torch.tensor([(scale * x, scale * y) for x, y in centers])
    x = sq2 * (0.5 * z + centers[torch.randint(len(centers),
                                               size=(num_samples,))])

    x = x.type(torch.float32).to('cpu')
    # x = torch.tensor(points).type(torch.float32).to(device)

    return x


# Pz = DiagGaussian(torch.tensor([0.0, 0.0]), torch.tensor([[1., 0.0], [0.0, 1.]]))
# z = torch.randn(10, 2)
# sample = Pz.sample((100,))

# x = get_batch(5000)
# plt.scatter(x[:, 0], x[:, 1])
# plt.show()
