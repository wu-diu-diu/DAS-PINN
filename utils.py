import numpy as np
import matplotlib.pyplot as plt
import torch
from IPython import display


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
                 figsize=(3.5, 2.5)):
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


# animator = Animator(xlabel='x', ylabel='y', legend=['sin', 'cos'], xlim=[0, 10], ylim=[-1.1, 1.1])
#
# # 模拟动态绘制
# for i in np.arange(0, 10, 0.1):
#     x = i
#     y = [np.sin(i), np.cos(i)]  # sin 和 cos 两条曲线
#     animator.add(x, y)
# plt.show()

