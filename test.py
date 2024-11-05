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


import deepxde as dde
import numpy as np
from deepxde.backend import tf

# 亥姆霍兹方程参数
k0 = 1.0  # 波数

# 定义 PDE
def pde(x, u):
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    du_yy = dde.grad.hessian(u, x, i=1, j=1)
    return du_xx + du_yy + k0**2 * u

# 定义几何域
geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 1])  # 1x1 square

# 边界条件
bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on_boundary: on_boundary)

# 数据设置
data = dde.data.PDE(geom, pde, [bc], num_domain=400, num_boundary=100)

# 神经网络模型
net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")

# PINN模型
model = dde.Model(data, net)

# 编译模型
model.compile("adam", lr=1e-3)

# 训练模型
model.train(iterations=20000)

# 可视化结果
import matplotlib.pyplot as plt

# 生成测试数据
x_test = np.linspace(0, 1, 100)
y_test = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x_test, y_test)
xy_test = np.vstack((X.flatten(), Y.flatten())).T

# 预测
u_pred = model.predict(xy_test)

# 绘制结果
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, u_pred.reshape(X.shape), levels=50, cmap='viridis')
plt.colorbar(label='u(x, y)')
plt.title('Solution of the Helmholtz Equation')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
