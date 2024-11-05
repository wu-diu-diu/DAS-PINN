import torch


def Helmholtz(x, model):
    n = 2
    k0 = n * torch.pi * 2
    u = model(x)
    # u = transform(x, u_inner)
    # u_boundary = model(x_boundary)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x[:, 0], x, torch.ones_like(u_x[:, 0]), create_graph=True)[0][:, 0]  ## (num,)
    u_yy = torch.autograd.grad(u_x[:, 1], x, torch.ones_like(u_x[:, 1]), create_graph=True)[0][:, 1]
    f = k0 ** 2 * torch.sin(k0 * x[:, 0]) * torch.sin(k0 * x[:, 1])
    return -u_yy - u_xx - k0 ** 2 * u.squeeze() - f, u


def transform(x, y):  ## 硬边界条件
    res = (x[:, 0:1] + 1) * (1 - x[:, 0:1]) * (x[:, 1:2] + 1) * (1 - x[:, 1:2])
    return res * y


def exact_solution(x):
    n = 2
    k0 = n * torch.pi * 2
    return torch.sin(k0 * x[:, 0:1]) * torch.sin(k0 * x[:, 1:2])


# x = torch.randn((10, 2), requires_grad=True)
# model = My_MLP(2, 3, 10, 1, "tanh")
# u = model(x)
# loss = Helmholtz(x, u)


# 设置超参数
# k = 1.0  # 波数
# num_points = 10000  # 训练点数量
# num_epochs = 5000  # 训练轮数
# learning_rate = 0.001  # 学习率
#
#
# # 定义神经网络结构
# class PINN(nn.Module):
#     def __init__(self):
#         super(PINN, self).__init__()
#         self.hidden1 = nn.Linear(2, 50)
#         self.hidden2 = nn.Linear(50, 50)
#         self.output = nn.Linear(50, 1)
#
#     def forward(self, x):
#         x = torch.tanh(self.hidden1(x))
#         x = torch.tanh(self.hidden2(x))
#         return self.output(x)
#
#
# # 生成训练数据
# def generate_data(num_points):
#     x = np.random.uniform(-1, 1, (num_points, 2))
#     return torch.tensor(x, dtype=torch.float32, requires_grad=True)
#
#
# # 定义损失函数
# def pde_loss(model, x):
#     u = model(x)
#     u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
#     u_xx = torch.autograd.grad(u_x[:, 0], x, torch.ones_like(u_x[:, 0]), create_graph=True)[0][:, 0]
#     u_yy = torch.autograd.grad(u_x[:, 1], x, torch.ones_like(u_x[:, 1]), create_graph=True)[0][:, 1]
#     z = (u_xx + u_yy + k ** 2 * u)
#     return torch.mean(z ** 2)
#
#
# # 训练网络
# def train(model, num_epochs, learning_rate):
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     for epoch in range(num_epochs):
#         optimizer.zero_grad()
#         x_train = generate_data(num_points)
#         loss = pde_loss(model, x_train)
#         loss.backward()
#         optimizer.step()
#         # if epoch % 1000 == 0:
#         print(f'Epoch {epoch}, Loss: {loss.item()}')
#
#
# # 初始化模型
# model = PINN()
# train(model, num_epochs, learning_rate)
#
# # 可视化结果
# x_test = np.linspace(-1, 1, 100)
# y_test = np.linspace(-1, 1, 100)
# X, Y = np.meshgrid(x_test, y_test)
# grid_points = np.vstack([X.ravel(), Y.ravel()]).T
# grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)
#
# with torch.no_grad():
#     U_pred = model(grid_points_tensor).numpy().reshape(X.shape)
#
# plt.figure(figsize=(8, 6))
# plt.contourf(X, Y, U_pred, levels=50, cmap='viridis')
# plt.colorbar()
# plt.title('Solution of Helmholtz Equation using PINN')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.xlim([-1, 1])
# plt.ylim([-1, 1])
# plt.show()
