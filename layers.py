import torch
import numpy as np
import torch.nn as nn


class BatchNorm(nn.Module):
    """
    input_size : features in input var
    :return normalize the input, output have same size with input
    """

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


class AffineCoupling(nn.Module):
    """
    input_size = input var dim
    split_size = split size of input var  将x分为x1和x2
    hidden_size = width of hidden layer
    n_hidden = depth of hidden layer
    cond_label_size = condition variable size
    net: input:split_size  output: 2*(input_size - split_size)
    :return same size with input
    """

    def __init__(self, input_size, split_size, hidden_size, n_hidden):
        super().__init__()
        self.log_beta = nn.Parameter(torch.zeros(input_size - split_size, dtype=torch.float32))
        self.input_size = input_size
        self.split_size = split_size
        net = [nn.Linear(split_size, hidden_size)]
        for _ in range(n_hidden - 1):
            net.append(nn.ReLU())
            net.append(nn.Linear(hidden_size, hidden_size))
        net.append(nn.ReLU())
        net.append(nn.Linear(hidden_size, 2 * (input_size - split_size)))
        self.net = nn.Sequential(*net)
        self.alpha = 0.6

    def forward(self, x):
        """
        x:tensor (N, d)
        y:tensor (N, cond_dim), condition_var
        """
        x1 = x[:, :self.split_size].view(-1, self.split_size)
        x2 = x[:, self.split_size:].view(-1, self.input_size - self.split_size)
        h = self.net(x1)
        s = h[:, :self.input_size - self.split_size]
        s = s.view(-1, self.input_size - self.split_size)
        t = h[:, self.input_size - self.split_size:]
        t = t.view(-1, self.input_size - self.split_size)
        ## krnet equation 15
        u2 = x2 * (1 + self.alpha * torch.tanh(s)) + torch.exp(torch.clip(self.log_beta, -5.0, 5.0)) * torch.tanh(t)
        ## computing jacobian matrix
        log_abs_det_jacobian = torch.log(1 + self.alpha * torch.tanh(s))
        log_abs_det_jacobian = log_abs_det_jacobian.sum(dim=1)
        # if self.split_size == (self.input_size / 2):
        #     return torch.cat([u2, x1], dim=1), log_abs_det_jacobian
        # else:
        #     return torch.cat([x1, u2], dim=1), log_abs_det_jacobian
        return torch.cat([x1, u2], dim=1), log_abs_det_jacobian

    def inverse(self, u):
        u1 = (u[:, :self.split_size]).view(-1, self.split_size)
        u2 = (u[:, self.split_size:]).view(-1, self.input_size - self.split_size)
        h = self.net(u1)
        s = h[:, :self.input_size - self.split_size]
        s = s.view(-1, self.input_size - self.split_size)
        t = h[:, self.input_size - self.split_size:]
        t = t.view(-1, self.input_size - self.split_size)

        x2 = (u2 - torch.exp(torch.clip(self.log_beta, -5.0, 5.0)) * torch.tanh(t)) / (1 + self.alpha * torch.tanh(s))
        log_abs_det_jacobian = -torch.log(1 + self.alpha * torch.tanh(s))
        log_abs_det_jacobian = log_abs_det_jacobian.sum(dim=1)
        return torch.cat([u1, x2], dim=-1), log_abs_det_jacobian


class Squeezing(nn.Module):
    """
    input_size: (N, dim)
    n_cut:  (N, dim) -> (N, dim- n_cut)
    self.x ??
    从后往前cut
    """

    def __init__(self, input_size, n_cut=1):
        super().__init__()
        # self.data_init = True
        self.input_size = input_size
        self.n_cut = n_cut
        self.x = None

    def forward(self, x):
        n_dim = x.shape[-1]
        if n_dim < self.n_cut:
            raise Exception()
        if self.input_size == n_dim:
            if self.input_size > self.n_cut:
                if self.x is not None:
                    raise Exception()
                else:
                    self.x = x[:, (self.input_size - self.n_cut):]  ## squeezing part
                    z = x[:, :(self.input_size - self.n_cut)]  ## squeezed part
            else:
                self.x = None
        elif n_dim <= self.n_cut:
            z = torch.cat([x, self.x], dim=-1)
            self.x = None
        else:
            cut = x[:, (n_dim - self.n_cut):]
            self.x = torch.cat([cut, self.x], dim=-1)
            z = x[:, :(n_dim - self.n_cut)]
        return z, 0

    def inverse(self, z):
        n_dim = z.shape[-1]
        if self.input_size == n_dim:
            n_start = self.input_size % self.n_cut
            if n_start == 0:
                n_start += self.n_cut
            self.x = z[:, n_start:]
            x = z[:, :n_start]
        else:
            x_length = self.x.shape[-1]
            if x_length < self.n_cut:
                raise Exception()

            cut = self.x[:, :self.n_cut]
            x = torch.cat([z, cut], dim=-1)
            if (x_length - self.n_cut) == 0:
                self.x = None
            else:
                self.x = self.x[:, self.n_cut:]
        return x, 0


class ActNorm(nn.Module):
    """
    y = scale * (x - bias) bias初始化为样本均值， scale初始化为1/（std + eps）
    """

    def __init__(self, input_size, scale=1.0, logscale_factor=3.0):
        super().__init__()
        self.scale = scale
        self.logscale_factor = logscale_factor
        self.data_init = True
        self.b = nn.Parameter(torch.zeros(1, input_size))
        self.register_buffer('b_init', torch.zeros(1, input_size))
        self.logs = nn.Parameter(torch.zeros(1, input_size))
        self.register_buffer('logs_init', torch.zeros(1, input_size))

    def forward(self, x):
        if not self.data_init:
            x_mean = torch.mean(x, 0, keepdinm=True)
            x_var = torch.mean(torch.square(x - x_mean), 0, keepdim=True)
            self.b_init = -x_mean
            self.logs_init = torch.log(self.scale / (torch.sqrt(x_var) + 1e-6)) / self.logscale_factor
            self.data_init = True
        y = x + self.b + self.b_init
        y = y * torch.exp(torch.clip(self.logs + self.logs_init, -5., 5.))

        log_abs_det_jacobian = torch.clip(self.logs + self.logs_init, -5., 5.)
        return y, log_abs_det_jacobian.expand_as(x).sum(dim=-1)

    def inverse(self, y, cond_y=None):
        x = y * torch.exp(-torch.clip(self.logs + self.logs_init, -5., 5.))
        x = x - (self.b + self.b_init)
        log_abs_det_jacobian = -torch.clip(self.logs + self.logs_init, -5., 5.)
        return x, log_abs_det_jacobian.expand_as(x).sum(dim=-1)

    def reset_data_initialization(self):
        self.data_init = False


class CDF_quadratic(nn.Module):
    """
    Nonlinear layer
    """

    def __init__(self, n_bins, input_dim, r=1.2, bound=50.0, **kwargs):
        super().__init__(**kwargs)
        assert n_bins % 2 == 0
        self.n_bins = n_bins
        self.input_dim = input_dim
        self.bound = bound
        self.r = r

        m = n_bins / 2  ## 16
        x1L = bound * (r - 1.0) / (np.power(r, m) - 1.0)  ## 50*0.2 / 1.2**16 -1 =

        index = torch.reshape(torch.arange(0, self.n_bins + 1, dtype=torch.float32), (-1, 1))
        index -= m  ## [-16,-15,-14,,,,,0,1,2,,16]
        xr = torch.where(index >= 0, (1. - torch.pow(r, index)) / (1. - r),
                         (1. - torch.pow(r, torch.abs(index))) / (1. - r))
        xr = torch.where(index >= 0, x1L * xr, -x1L * xr)
        xr = torch.reshape(xr, (-1, 1))  ## [33,1] -50 - 50
        xr = (xr + bound) / 2.0 / bound  ## [33,1] 0-1的tensor

        self.x1L = x1L / 2.0 / bound
        ## mesh [33,1] 0-1的tensor
        mesh = torch.cat([torch.reshape(torch.tensor([0.0]), (-1, 1)), torch.reshape(xr[1:-1, 0], (-1, 1)),
                          torch.reshape(torch.tensor([1.0]), (-1, 1))], 0)
        self.register_buffer('mesh', mesh)
        elmt_size = torch.reshape(self.mesh[1:] - self.mesh[:-1], (-1, 1))
        self.register_buffer('elmt_size', elmt_size)  ## [32, 1] 去掉了0和1 而且对称
        self.p = nn.Parameter(torch.zeros(self.n_bins - 1, input_dim))

    def forward(self, x, t=None):
        self._pdf_normalize()
        # rescale such points in [-bound, bound] will be mapped to [0,1]
        x = (x + self.bound) / 2.0 / self.bound

        # cdf mapping
        x, logdet = self._cdf(x)

        # maps [0,1] back to [-bound, bound]
        x = x * 2.0 * self.bound - self.bound
        return x, logdet

    def inverse(self, z, t=None):
        self._pdf_normalize()
        # rescale such points in [-bound, bound] will be mapped to [0,1]
        x = (z + self.bound) / 2.0 / self.bound

        # cdf mapping
        x, logdet = self._cdf_inv(x)

        # maps [0,1] back to [-bound, bound]
        x = x * 2.0 * self.bound - self.bound
        return x, logdet

    def _pdf_normalize(self):
        # peicewise pdf
        p0 = torch.ones((1, self.input_dim), dtype=torch.float32, device=self.mesh.device)
        self.pdf = p0
        px = torch.exp(self.p) * (self.elmt_size[:-1] + self.elmt_size[1:]) / 2.0  ## 31*10
        px = (1 - self.elmt_size[0]) / torch.sum(px, 0, keepdim=True)  ##1*10
        px = px * torch.exp(self.p)  ##31*10
        self.pdf = torch.concat([self.pdf, px], 0)  ## 32*10
        self.pdf = torch.concat([self.pdf, p0], 0)  ## 33*10

        # probability in each element
        cell = (self.pdf[:-1, :] + self.pdf[1:, :]) / 2.0 * self.elmt_size  ## 32*10
        # CDF - contribution from previous elements.
        r_zeros = torch.zeros((1, self.input_dim), dtype=torch.float32, device=self.mesh.device)
        self.F_ref = r_zeros
        for i in range(1, self.n_bins):  ## 1-31
            tp = torch.sum(cell[:i, :], 0, keepdim=True)  ## 将cell的
            self.F_ref = torch.concat([self.F_ref, tp], 0)  ## 32*10 每个维度都是0-10分成了32份

    def _cdf(self, x):
        x_sign = torch.sign(x - 0.5)
        m = torch.floor(torch.log(torch.abs(x - 0.5) * (self.r - 1) / self.x1L + 1.0) / np.log(self.r))
        k_ind = torch.where(x_sign >= 0, self.n_bins / 2 + m, self.n_bins / 2 - m - 1)
        k_ind = k_ind.to(dtype=torch.int64)
        cover = torch.where(k_ind * (k_ind - self.n_bins + 1) <= 0, 1.0, 0.0)

        # print('k_ind', k_ind)
        k_ind = torch.where(k_ind < 0, 0 * k_ind, k_ind)
        k_ind = torch.where(k_ind > (self.n_bins - 1), (self.n_bins - 1) * torch.ones_like(k_ind), k_ind)

        # print(self.pdf[:,0].shape)

        # print(k_ind[:,0])
        v1 = torch.reshape(torch.gather(self.pdf[:, 0], 0, k_ind[:, 0]), (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(torch.gather(self.pdf[:, i], 0, k_ind[:, i]), (-1, 1))
            v1 = torch.concat([v1, tp], 1)

        v2 = torch.reshape(torch.gather(self.pdf[:, 0], 0, k_ind[:, 0] + 1), (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(torch.gather(self.pdf[:, i], 0, k_ind[:, i] + 1), (-1, 1))
            v2 = torch.concat([v2, tp], 1)

        xmodi = torch.reshape(x[:, 0] - torch.gather(self.mesh[:, 0], 0, k_ind[:, 0]), (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(x[:, i] - torch.gather(self.mesh[:, 0], 0, k_ind[:, i]), (-1, 1))
            xmodi = torch.concat([xmodi, tp], 1)

        h_list = torch.reshape(torch.gather(self.elmt_size[:, 0], 0, k_ind[:, 0]), (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(torch.gather(self.elmt_size[:, 0], 0, k_ind[:, i]), (-1, 1))
            h_list = torch.concat([h_list, tp], 1)

        F_pre = torch.reshape(torch.gather(self.F_ref[:, 0], 0, k_ind[:, 0]), (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(torch.gather(self.F_ref[:, i], 0, k_ind[:, i]), (-1, 1))
            F_pre = torch.concat([F_pre, tp], 1)

        y = torch.where(cover > 0, F_pre + xmodi ** 2 / 2.0 * (v2 - v1) / h_list + xmodi * v1, x)

        dlogdet = torch.where(cover > 0, xmodi * (v2 - v1) / h_list + v1, torch.ones_like(cover))
        dlogdet = torch.sum(torch.log(dlogdet), dim=[1])

        return y, dlogdet

    # inverse of the cdf
    def _cdf_inv(self, y):
        xr = torch.broadcast_to(self.mesh, [self.n_bins + 1, self.input_dim])
        yr1, _ = self._cdf(xr)

        p0 = torch.zeros((1, self.input_dim), device=self.mesh.device, dtype=torch.float32)
        p1 = torch.ones((1, self.input_dim), device=self.mesh.device, dtype=torch.float32)
        yr = torch.concat([p0, yr1[1:-1, :], p1], 0)

        k_ind = torch.searchsorted((yr.T).contiguous(), (y.T).contiguous(), right=True)
        k_ind = torch.transpose(k_ind, 0, 1)
        k_ind = k_ind.to(dtype=torch.int64)
        k_ind -= 1

        cover = torch.where(k_ind * (k_ind - self.n_bins + 1) <= 0, 1.0, 0.0)

        k_ind = torch.where(k_ind < 0, 0, k_ind)
        k_ind = torch.where(k_ind > (self.n_bins - 1), self.n_bins - 1, k_ind)

        c_cover = torch.reshape(cover[:, 0], (-1, 1))

        v1 = torch.where(c_cover > 0, torch.reshape(torch.gather(self.pdf[:, 0], 0, k_ind[:, 0]), (-1, 1)),
                         -1. * torch.ones_like(c_cover))
        for i in range(1, self.input_dim):
            c_cover = torch.reshape(cover[:, i], (-1, 1))
            tp = torch.where(c_cover > 0, torch.reshape(torch.gather(self.pdf[:, i], 0, k_ind[:, i]), (-1, 1)),
                             -1.0 * torch.ones_like(c_cover))
            v1 = torch.concat([v1, tp], 1)

        c_cover = torch.reshape(cover[:, 0], (-1, 1))
        v2 = torch.where(c_cover > 0, torch.reshape(torch.gather(self.pdf[:, 0], 0, k_ind[:, 0] + 1), (-1, 1)),
                         -2.0 * torch.ones_like(c_cover))
        for i in range(1, self.input_dim):
            c_cover = torch.reshape(cover[:, i], (-1, 1))
            tp = torch.where(c_cover > 0, torch.reshape(torch.gather(self.pdf[:, i], 0, k_ind[:, i] + 1), (-1, 1)),
                             -2.0 * torch.ones_like(c_cover))
            v2 = torch.concat([v2, tp], 1)

        ys = torch.reshape(y[:, 0] - torch.gather(yr[:, 0], 0, k_ind[:, 0]), (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(y[:, i] - torch.gather(yr[:, i], 0, k_ind[:, i]), (-1, 1))
            ys = torch.concat([ys, tp], 1)

        xs = torch.reshape(torch.gather(xr[:, 0], 0, k_ind[:, 0]), (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(torch.gather(xr[:, i], 0, k_ind[:, i]), (-1, 1))
            xs = torch.concat([xs, tp], 1)

        h_list = torch.reshape(torch.gather(self.elmt_size[:, 0], 0, k_ind[:, 0]), (-1, 1))
        for i in range(1, self.input_dim):
            tp = torch.reshape(torch.gather(self.elmt_size[:, 0], 0, k_ind[:, i]), (-1, 1))
            h_list = torch.concat([h_list, tp], 1)

        tp = 2.0 * ys * h_list * (v2 - v1)
        tp += v1 * v1 * h_list * h_list
        tp = torch.sqrt(tp) - v1 * h_list
        tp = torch.where(torch.abs(v1 - v2) < 1.0e-6, ys / v1, tp / (v2 - v1))
        tp += xs

        x = torch.where(cover > 0, tp, y)

        tp = 2.0 * ys * h_list * (v2 - v1)
        tp += v1 * v1 * h_list * h_list
        tp = h_list / torch.sqrt(tp)

        dlogdet = torch.where(cover > 0, tp, torch.ones_like(cover))
        dlogdet = torch.sum(torch.log(dlogdet), dim=[1])

        return x, dlogdet


class scale_and_CDF(nn.Module):
    """
    Combination of scale-bias layer and non-linear CDF layer
    Args:
        input_size, int, input vector dimension
        n_bins, int, num of bins of non-linear CDF layer
    """

    def __init__(self, input_size, n_bins=16, r=1.2, bound=50.0) -> None:
        super().__init__()
        self.scale_layer = ActNorm(input_size)
        self.cdf_layer = CDF_quadratic(n_bins, input_size, r, bound)

    def forward(self, x):
        z = x
        log_det = 0
        z, tmp_log_det = self.scale_layer.forward(z)
        log_det = log_det + tmp_log_det
        z, tmp_log_det = self.cdf_layer.forward(z)
        log_det = log_det + tmp_log_det
        return z, log_det

    def inverse(self, z):
        print(z.shape)
        x = z
        log_det = 0
        x, tmp_log_det = self.cdf_layer.inverse(x)
        log_det = log_det + tmp_log_det

        x, tmp_log_det = self.scale_layer.inverse(x)
        log_det = log_det + tmp_log_det
        return x, log_det


class W_LU(nn.Module):
    """
    rotation layer
    """

    def __init__(self, input_size):
        super().__init__()
        self.data_init = True
        self.input_size = input_size
        self.LU = nn.Parameter(torch.zeros(input_size, input_size, dtype=torch.float32))
        self.register_buffer('LU_init', torch.eye(input_size, input_size, dtype=torch.float32))
        self.register_buffer('ones_mat', torch.ones(input_size, input_size, dtype=torch.float32))  ## 全为1的矩阵
        self.register_buffer('I', torch.eye(input_size))

    def forward(self, x):
        LU = self.LU_init + self.LU  ## 确保第一次运行时LU是一个单位阵
        U = torch.triu(LU)  ## 得到LU矩阵的上三角阵
        U_diag = torch.diagonal(U)  ## 得到U的对角线元素并展开成1行
        U_mask = torch.triu(self.ones_mat)  ## onesmat是一个全为1的矩阵获得其上三角阵是为了控制 U 只更新上三角
        U = ((1 - U_mask) * U).detach() + U_mask * U  ## U_mask上三角为1下三角为0 这一步将U的下三角不包括对角线切断其计算图避免反向传播更新梯度
        ## 下面L同理 L矩阵是下三角矩阵且对角线元素均为1
        L = torch.tril(self.I + LU) - torch.diagonal(LU).diag()  ## 得到LU下三角阵的同时，令下三角阵的对角线元素为1
        L_mask = torch.tril(self.ones_mat) - self.I
        L = ((1 - L_mask) * L).detach() + L_mask * L  ## 将L的上三角包括对角线元素均冻结不让其更新梯度

        x = x.T
        x = torch.matmul(U, x)
        x = torch.matmul(L, x)
        x = x.T

        log_abs_det_jacobian = torch.log(torch.abs(U_diag))
        return x, log_abs_det_jacobian.expand_as(x).sum(dim=-1)

    def inverse(self, x):
        LU = self.LU_init + self.LU  ## 确保第一次运行时LU是一个单位阵
        U = torch.triu(LU)  ## 得到LU矩阵的上三角阵
        U_diag = torch.diagonal(U)  ## 得到U的对角线元素并展开成1行
        U_mask = torch.triu(self.ones_mat)  ## onesmat是一个全为1的矩阵获得其上三角阵是为了控制 U 只更新上三角
        U = ((1 - U_mask) * U).detach() + U_mask * U  ## U_mask上三角为1下三角为0 这一步将U的下三角不包括对角线切断其计算图避免反向传播更新梯度
        ## 下面L同理 L矩阵是下三角矩阵且对角线元素均为1
        L = torch.tril(self.I + LU) - torch.diagonal(LU).diag()  ## 得到LU下三角阵的同时，令下三角阵的对角线元素为1
        L_mask = torch.tril(self.ones_mat) - self.I
        L = ((1 - L_mask) * L).detach() + L_mask * L  ## 将L的上三角包括对角线元素均冻结不让其更新梯度

        x = torch.transpose(x, 0, 1)
        x = torch.matmul(torch.inverse(L), x)
        x = torch.matmul(torch.inverse(U), x)
        x = torch.transpose(x, 0, 1)
        log_abs_det_jacobian = -torch.log(torch.abs(U_diag))
        return x, log_abs_det_jacobian.expand_as(x).sum(dim=-1)

    def reset_data_initialization(self):
        self.data_init = False


class flow_mapping(nn.Module):
    """
    input_size: 输入变量的特征数
    n_depth: scale_layer+affinecoupling 的层数
    n_split_at: feature分成两份的比例数
    n_hidden: affinecoupling 中隐藏层的个数
    n_width: affinecoupling 中 隐藏层的神经元个数
    flowing_map:uesless
    n_bins, r， bound均是cdf参数
    """

    def __init__(self, input_size, n_depth, n_split_at, n_hidden=2, n_width=32, n_bins=32, r=1.2, bound=50.0):
        super().__init__()
        self.n_depth = n_depth
        self.n_split_at = n_split_at
        self.n_width = n_width
        self.n_bins = n_bins
        self.n_hidden = n_hidden
        assert n_depth % 2 == 0
        self.input_size = input_size
        self.scale_layers = []
        self.affine_layers = []
        sign = -1
        for i in range(self.n_depth):
            self.scale_layers.append(ActNorm(input_size))
            sign *= -1
            i_split_at = (self.n_split_at * sign + self.input_size) % self.input_size  ## i 交替出现为8 2 8 2 从而达到仿射耦合层交叉耦合的效果
            self.affine_layers.append(AffineCoupling(input_size,
                                                     i_split_at,
                                                     hidden_size=self.n_width,
                                                     n_hidden=self.n_hidden))
        self.scale_layers = nn.ModuleList(self.scale_layers)
        self.affine_layers = nn.ModuleList(self.affine_layers)
        self.cdf_layer = CDF_quadratic(self.n_bins, input_size - self.n_split_at, r, bound)

    def forward(self, x):
        log_det = 0
        z = x
        for i in range(self.n_depth):
            z, tmp_log_det = self.scale_layers[i](z)
            log_det += tmp_log_det
            z, tmp_log_det = self.affine_layers[i](z)
            log_det += tmp_log_det
            z = torch.flip(z, (-1,))
        i_split_at = self.n_split_at
        z1 = z[:, :i_split_at]
        z2 = z[:, i_split_at:]
        z2, tmp_log_det = self.cdf_layer(z2)
        log_det += tmp_log_det
        z = torch.cat([z1, z2], dim=-1)
        return z, log_det

    def inverse(self, z):
        x = z
        log_det = 0
        i_split_at = self.n_split_at
        x1 = x[:, :i_split_at]
        x2 = x[:, i_split_at:]
        x2, tmp_log_det = self.cdf_layer.inverse(x2)
        log_det = log_det + tmp_log_det
        x = torch.cat([x1, x2], dim=-1)
        for i in reversed(range(self.n_depth)):
            x = torch.flip(x, (-1,))
            x, tmp_log_det = self.affine_layers[i].inverse(x)
            log_det = log_det + tmp_log_det
            x, tmp_log_det = self.scale_layers[i].inverse(x)
            log_det = log_det + tmp_log_det
        return x, log_det


class realNVP_KR_CDF(nn.Module):
    """
    input_size: the features of input tensor
    n_step:每个stage要对input减去的维度，也就是文中的number K of partition
    n_depth: the layer depth of scale and bias + affinecouplig 也就是文中的L
    width: the hidden size in affinecoupling layer
    n_hidden: 放射耦合层中隐藏层的层数
    n_bins: 和cdf层相关
    """

    def __init__(self, input_size, n_step, n_depth, width, n_hidden, n_bins=32, r=1.2, bound=50.0, shrink_rate=1.0,
                 rotation=True):
        super().__init__()
        self.shrink_rate = shrink_rate
        self.input_size = input_size
        self.n_step = n_step
        self.n_hidden = n_hidden

        self.n_stage = input_size // n_step  ## 减去维度这个过程循环的次数也就是网络中外面的大循环
        if n_step == 1:  ## 这里为什么要针对n_step的情况对stage减1
            self.n_stage = self.n_stage - 1
        self.rotation = rotation
        self.n_rotation = 1  ##
        if rotation:
            self.rotations = nn.ModuleList([W_LU(input_size)])
            for i in range(self.rotation - 1):  ## 括号里是0则不会添加rotation层 故n_rotation = 1的话就不会添加新的rotation
                self.rotations.append(W_LU(input_size))
        self.flow_mapping = []
        n_width = width
        for i in range(self.n_stage):
            n_split_at = input_size - (i + 1) * n_step  ## n_split_at就是一个将feature分开的比例
            self.flow_mapping.append(
                flow_mapping(input_size - i * n_step,
                             n_depth,
                             n_split_at,
                             n_hidden,
                             n_width=n_width,
                             n_bins=n_bins,
                             r=r,
                             bound=bound))
            n_width = int(n_width * self.shrink_rate)  ## 仿射耦合层中的hidden_size随着外循环的进行在衰减
        self.flow_mapping = nn.ModuleList(self.flow_mapping)
        self.squeezing_layer = Squeezing(input_size, n_step)
        self.scale_CDF = scale_and_CDF(input_size, n_bins, r, bound)

    def forward(self, x):
        z = x
        log_det = 0
        for i in range(self.n_stage - 1):
            if self.rotation and i < self.n_rotation:
                z, tmp_log_det = self.rotations[i](z)
                log_det += tmp_log_det
            z, tmp_log_det = self.flow_mapping[i](z)
            log_det += tmp_log_det
            z, _ = self.squeezing_layer(z)
        z, tmp_log_det = self.flow_mapping[self.n_stage - 1](z)
        log_det += tmp_log_det

        if self.squeezing_layer.x is not None:
            z = torch.cat([z, self.squeezing_layer.x], dim=-1)
        del self.squeezing_layer.x  ## 显示的删除了这行代码
        self.squeezing_layer.x = None
        z, tmp_log_det = self.scale_CDF(z)
        log_det += tmp_log_det
        return z, log_det

    def inverse(self, z):
        x = z
        log_det = 0
        x, tmp_log_det = self.scale_CDF.inverse(x)
        log_det = log_det + tmp_log_det
        # Start squeeze layer operation
        n_start = self.input_size % self.n_step
        if self.n_step == 1:
            n_start = 1
        n_start = n_start + self.n_step
        store_x_size = self.input_size - n_start
        if self.n_stage > 1:
            self.squeezing_layer.x = x[:, n_start:]
            x = x[:, :n_start]
        # End
        x, tmp_log_det = self.flow_mapping[self.n_stage - 1].inverse(x)
        log_det = log_det + tmp_log_det
        for i in reversed(range(self.n_stage - 1)):
            x, _ = self.squeezing_layer.inverse(x)
            x, tmp_log_det = self.flow_mapping[i].inverse(x)
            log_det = log_det + tmp_log_det
            if self.rotation and i < self.n_rotation:
                x, tmp_log_det = self.rotations[i].inverse(x)
                log_det = log_det + tmp_log_det
        del self.squeezing_layer.x
        self.squeezing_layer.x = None
        # x, tmp_log_det = self.scale_CDF.inverse(x)
        # log_det = log_det + tmp_log_det
        return x, log_det


class KRnet(nn.Module):
    def __init__(self, base_dist, input_size, n_step, n_depth, width, n_hidden, n_bins=32, r=1.2, bound=50.0,
                 shrink_rate=1.0, rotation=True):
        super().__init__()
        self.net = realNVP_KR_CDF(input_size, n_step, n_depth, width, n_hidden, n_bins, r, bound, shrink_rate, rotation)
        self.base_dist = base_dist
        # self.device = device

    def forward(self, x):
        z, log_det = self.net.forward(x)
        return z, log_det

    def inverse(self, z):
        x, log_det = self.net.inverse(z)
        return x, log_det

    def log_prob(self, x):
        z, log_det = self.forward(x)
        base_log_prob = self.base_dist.log_prob(z)
        log_prob = base_log_prob.to(device=x.device) + log_det
        return log_prob

    def sample(self, n_sample):
        base_samples = self.base_dist.sample(n_sample)
        samples, _ = self.inverse(base_samples)
        return samples
