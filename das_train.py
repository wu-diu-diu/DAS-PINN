import torch
from layers import *
from utils import Animator
import matplotlib.pyplot as plt
import numpy as np
import argparse
from utils import My_MLP, get_valid_data, get_train_data
from equation import Helmholtz, exact_solution
import os
from distributions import DiagGaussian
from das_model import DAS


def set_seed(seed=42):
    np.random.seed(seed)  # Numpy 随机种子
    torch.manual_seed(seed)  # PyTorch 随机种子（CPU）
    torch.cuda.manual_seed(seed)  # PyTorch 随机种子（GPU）


def train(args, save_path, model, train_data, loss_save_path):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lambda_data, lambda_exact = 1e-4, 0
    # animator = Animator(xlabel='epoch', ylabel='loss', xlim=[1, args.iter + 1], ylim=[1e-3, 1e-1],
    #                     legend='train_loss')  ## 添加loss曲线图参数
    for i in range(1, args.iter + 1):
        optimizer.zero_grad()
        pde_res, u_pred = Helmholtz(train_data, model)
        loss_data = torch.mean(pde_res ** 2)
        loss_exact = criterion(exact_solution(train_data), u_pred)
        loss = lambda_data * loss_data + lambda_exact * loss_exact
        loss.backward()
        optimizer.step()
        # if i % 100 == 0:
        #     animator.add(i, [loss.detach().cpu().numpy()])
        print(f"Epoch: {i}, loss: {loss.item():.7f}")
    # plt.savefig(loss_save_path, bbox_inches='tight')
    # plt.show()
    torch.save(model.state_dict(), save_path)


def inference(model, save_path, train_data, image_save_path, device):
    model.load_state_dict(torch.load(save_path))
    model.eval()
    with torch.no_grad():
        x_test = np.linspace(-1, 1, 100)
        y_test = np.linspace(-1, 1, 100)
        X, Y = np.meshgrid(x_test, y_test)
        grid_points = np.vstack([X.ravel(), Y.ravel()]).T
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)

        u_pred = model(grid_points_tensor).cpu().numpy().reshape(X.shape)
        exact = exact_solution(grid_points_tensor).cpu().numpy().reshape(X.shape)
        res = np.abs(u_pred - exact)
        fig, axs = plt.subplots(1, 4, figsize=(24, 6))
        pc_exact = axs[0].pcolormesh(X, Y, exact, shading='auto', cmap='viridis')
        axs[0].set_title('Exact Solution')
        axs[0].set_xlabel('X-axis')
        axs[0].set_ylabel('Y-axis')
        plt.colorbar(pc_exact, ax=axs[0])

        # 绘制模型输出的热图
        pc_model = axs[1].pcolormesh(X, Y, u_pred, shading='auto', cmap='viridis')
        axs[1].set_title('Model Output')
        axs[1].set_xlabel('X-axis')
        axs[1].set_ylabel('Y-axis')
        plt.colorbar(pc_model, ax=axs[1])

        pc_res = axs[2].pcolormesh(X, Y, res, shading='auto', cmap='viridis')
        axs[2].set_title('residual')
        axs[2].set_xlabel('X-axis')
        axs[2].set_ylabel('Y-axis')
        plt.colorbar(pc_res, ax=axs[2])

        scatter = axs[3].scatter(train_data[:, 0].cpu().detach().numpy(), train_data[:, 1].cpu().detach().numpy(),
                                 alpha=0.8, s=1)
        axs[3].set_title('train_data')
        axs[3].set_xlabel('X-axis')
        axs[3].set_ylabel('Y-axis')
        plt.savefig(image_save_path, bbox_inches='tight')
        plt.tight_layout()
        plt.show()


def main(args):
    device = torch.device("cuda:0")
    set_seed()
    prior_dist = DiagGaussian(torch.tensor([0.0, 0.0], device=device),
                              torch.tensor([[1., 0.0], [0.0, 1.]], device=device))
    x, x_boundary = get_train_data(args.input_size, args.num_data, args.num_bd_data,
                                   bd=args.data_bd, device=device)
    train_data = x
    valid_data = get_valid_data(args.data_bd, args.num_valid_data, device=device)
    u_true_mesh = exact_solution(valid_data)
    model = DAS(args, device=device, prior_dist=prior_dist)
    image_save_path = os.path.join(args.images_dir,
                                   "das_stage{}_{}_{}_{}.pth".format(args.stage,
                                                                     args.flow_epoch,
                                                                     args.pde_epoch, args.num_data))
    loss_save_path = os.path.join(args.loss_dir,
                                  "das_stage{}_{}_{}_{}.pth".format(args.stage,
                                                                     args.flow_epoch,
                                                                     args.pde_epoch, args.num_data))
    if args.mode == 'train':
        model.train(train_data, valid_data)
    elif args.mode == 'test':
        inference(pde_model, param_save_path, train_data, image_save_path, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--input_size", type=int, default=2)  ## 数据维度
    parser.add_argument("--num_data", type=int, default=10000)  ## 数据个数 边界和内部各num个数据
    parser.add_argument("--num_bd_data", type=int, default=1000)  ## 数据个数 边界和内部各num个数据
    parser.add_argument("--num_valid_data", type=int, default=256)  ## 验证数据，256 *256 的网格
    parser.add_argument("--num_sample", type=int, default=10000)  ## 训练flow后采样数据的个数
    parser.add_argument("--data_bd", type=int, default=1)  ## 数据范围 [-bd， bd]²

    parser.add_argument("--flow_epoch", type=int, default=4000)  ## 数据个数 边界和内部各num个数据
    parser.add_argument("--pde_epoch", type=int, default=3000)  ## 数据个数 边界和内部各num个数据
    parser.add_argument("--stage", type=int, default=5)  ## 数据个数 边界和内部各num个数据

    parser.add_argument("--flow_n_step", type=int, default=1)  ## KRnet框架下数据每次衰减的维度
    parser.add_argument("--flow_n_depth", type=int, default=10)  ## KRnet中affine堆积层数
    parser.add_argument("--flow_width", type=int, default=32)  ## krnet中affine隐藏层神经元数
    parser.add_argument("--flow_n_hidden", type=int, default=5)  ## krnet中affine层中隐藏层个数

    parser.add_argument("--pde_output_size", type=int, default=1)  ## 数据维度
    parser.add_argument("--pde_width", type=int, default=150)  ## PDEmodel中隐藏层神经元数
    parser.add_argument("--pde_n_hidden", type=int, default=2)  ## PDEmodel中中隐藏层层数
    parser.add_argument("--pde_activation", type=str, default='sin')  ## PDEmodel的激活函数
    parser.add_argument("--param_path", default="param")
    parser.add_argument("--images_dir", default="./images")
    parser.add_argument("--loss_dir", default="./loss_curve")
    args = parser.parse_args()
    main(args)
