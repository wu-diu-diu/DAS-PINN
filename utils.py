import os.path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from equation import transform
from PIL import Image
import os


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)


class Animator:
    """在动画中绘制数据"""
    """
    add函数： x：int y: 可接受多个参数，类型为ndarray
    """

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(7, 5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []  ## 没有提供legend也初始化为一个列表，防止后续报错
        plt.rcParams['figure.figsize'] = figsize  ## 控制生成图像的大小
        self.fig, self.axes = plt.subplots(nrows, ncols)  ## 子图的行列数
        if nrows * ncols == 1:
            self.axes = [self.axes, ]  ## 如果只有一个子图，也将axes变为列表
        # 使用lambda函数捕获参数
        self.config_axes = lambda: self.set_axes(  ## 使用lambda封装设置函数轴的操作可以简化代码，调用self.config_axes即可
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts  ## fmts中的线条格式会按顺序绘制

    def set_axes(self, axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """设置matplotlib的轴"""
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()

    def add(self, x, y):
        """向图表中添加多个数据点"""
        if not hasattr(y, "__len__"):  ## y是否有__len__属性
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]  ## 在一个大列表中创建n个空列表
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            self.X[i].append(a)
            self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        # display.display(self.fig)
        # display.clear_output(wait=True)
        plt.pause(0.1)  # 短暂停止以更新图像
        plt.draw()


def save_checkpoint(model, optimizer, epoch, loss, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filename)


def load_checkpoint(filename, model, optimizer):
    print(f"loading ckpt from :{filename}")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint["epoch"]
    return start_epoch


class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(4 * torch.pi * input)  ## 这里为什么乘以了30


class My_MLP(nn.Module):
    def __init__(self, input_size, num_hidden, hidden_size, output_size, acti_name):
        super().__init__()
        assert isinstance(acti_name, str), f"type of acti_name should be string while input is {type(acti_name)}"
        acti = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "leaky relu": nn.LeakyReLU(0.01),
            "sin": Sine()
        }
        assert acti_name in acti.keys(), f"{[act for act in acti.keys()]} are used while input is {acti_name}"
        self.num_hidden = num_hidden
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden - 1)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.acti = acti[acti_name]
        # self._initialize_weight()

    def forward(self, x):
        inputs = x
        x = self.acti(self.input_layer(x))
        for i in range(self.num_hidden - 1):
            x = self.acti(self.hidden_layers[i](x))
        x = self.output_layer(x)
        x = transform(inputs, x)  ## 硬边界条件
        return x

    # def _initialize_weight(self):
    #     for layer in [self.input_layer] + self.hidden_layers + [self.output_layer]:
    #         if isinstance(layer, nn.Linear):
    #             nn.init.normal_(layer.weight, std=1, mean=0)
    #             nn.init.zeros_(layer.bias)


def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)


def gen_square_domain(ndim, n_train, bd=1.0, unitcube=False, hyperuniform=False):
    """
    random samples in square domain  default to [-1, 1]^d (len_edge=2)
    """

    # subfunction: generate samples uniformly at random in a ball
    def gen_nd_ball(n_sample, n_dim):
        x_g = np.random.randn(n_sample, n_dim)
        u_number = np.random.rand(n_sample, 1)  ## 生成n个0-1之间的均匀随机数
        x_normalized = x_g / np.sqrt(np.sum(x_g ** 2, axis=1, keepdims=True))  ## 将n个多维样本规范化，变为单位向量
        x_sample = (u_number ** (1 / n_dim) * x_normalized).astype(np.float32)  ## 将规范化的样本乘以u_number**(1/n_dim)
        ## 使其位置在球内
        return x_sample

    if not unitcube:
        # if hyperuniform, half data points drawn from a unit ball and half from uniform distribution
        if hyperuniform:
            n_corner = n_train // 2
            # n_circle = (1 - np.exp(-ndim/10.0)) * n_train
            n_circle = n_train - n_corner
            # generate samples uniformly at random from a unit ball
            x_circle = gen_nd_ball(n_circle, ndim)
            # most of these samples are lies in the corner of a hypercube
            x_corner = np.random.uniform(-bd, bd, [n_corner, ndim]).astype(np.float32)
            x = np.concatenate((x_circle, x_corner), axis=0)

        else:
            x = np.random.uniform(-bd, bd, [n_train, ndim]).astype(np.float32)

    else:
        x = np.random.uniform(0, 1, [n_train, ndim]).astype(np.float32)
    return x


def gen_square_domain_boundary(ndim, n_train, bd=1.0, unitcube=False):
    """
    random samples on the boundary of square domain [-1,1]^2 (default)
    """
    if ndim != 2:
        raise ValueError('ndim must be 2 for boundary of square domain')

    # boundary of square domain are four edges: top (0), bottom (1), left (2), and right (3).
    edges = [0, 1, 2, 3]
    x_boundary = np.zeros((n_train, ndim), dtype='float32')
    for i in range(n_train):
        which_edge = np.random.choice(edges)
        if which_edge == 0:
            point_x = np.random.uniform(-bd, bd)
            point_y = bd
        elif which_edge == 1:
            point_x = np.random.uniform(-bd, bd)
            point_y = -bd
        elif which_edge == 2:
            point_x = -bd
            point_y = np.random.uniform(-bd, bd)
        elif which_edge == 3:
            point_x = bd
            point_y = np.random.uniform(-bd, bd)

        point = np.array([point_x, point_y])
        x_boundary[i, :] = point

    if unitcube:
        x_boundary = 1 / (2 * bd) * x_boundary + 0.5
        return x_boundary

    return x_boundary


def get_train_data(n_dim, n_sample, num_bd_data, bd, device):
    train_data = gen_square_domain(n_dim, n_sample, bd=bd)
    train_data_boundary = gen_square_domain_boundary(n_dim, num_bd_data, bd=bd)
    return (torch.tensor(train_data, dtype=torch.float32, requires_grad=True).to(device),
            torch.tensor(train_data_boundary, dtype=torch.float32, requires_grad=True).to(device))


def get_valid_data(bd, num_data, device):
    # 生成坐标
    x = np.linspace(-bd, bd, num_data)  # 在[-1, 1]区间生成256个点
    y = np.linspace(-bd, bd, num_data)  # 在[-1, 1]区间生成256个点

    # 创建网格
    X, Y = np.meshgrid(x, y)

    # 将网格点组合成一个二维数组
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    grid_points = torch.tensor(grid_points, dtype=torch.float32).to(device)
    return grid_points


def exact_solution(x):
    n = 2
    k0 = n * torch.pi * 2
    return torch.sin(k0 * x[:, 0:1]) * torch.sin(k0 * x[:, 1:2])


def projection_onto_infunitball(x):
    """
    projection onto the infinity ball ||x||_inf <=1
    y = prox(x), then
    y = 1, if x >= 1; x, if |x| < 1; -1, if x <= -1.

    Args:
    -----
        x: data points
        probsetup: problem type

    Returns:
    --------
        projection onto infnity ball
    """
    assert len(x.shape) == 2
    n_sample = x.shape[0]
    boundary_idx = []

    for k in range(n_sample):
        # data point on the boundary
        if (x[k, :] > 1).any() or (x[k, :] < -1).any():
            # projection onto boundary
            boundary_idx.append(k)
            x[k, :] = np.sign(x[k, :]) * np.minimum(np.abs(x[k, :]), 1)

    x_boundary = x[boundary_idx, :]
    x = np.delete(x, boundary_idx, axis=0)

    return x, x_boundary


def get_images():
    image_dir = "images"
    res_image_files = [f"res_{i}.png" for i in range(1, 5)]
    num_images = len(res_image_files)
    fig, axs = plt.subplots(1, num_images, figsize=(16, 5))  # 创建 1 行 num_images 列的子图
    for i, image_file in enumerate(res_image_files):
        image_path = os.path.join(image_dir, image_file)

        # 打开图片
        img = Image.open(image_path)

        # 显示图片
        axs[i].imshow(img)
        axs[i].axis('off')  # 关闭坐标轴
        axs[i].set_title(f'stage {i + 1}')  # 设置标题
    plt.tight_layout()  # 调整子图间距
    plt.savefig(os.path.join(image_dir, "res_all.png"))
    plt.show()  # 显示图像


# get_images()

