import os.path

from layers import *
from distributions import DiagGaussian, original_dist
import torch.optim as optim
import argparse
from utils import *
import matplotlib.pyplot as plt
import time


def train_model(model, save_path, train_data, loss_save_path, ckpt_save_path):
    min_loss = float("inf")  ## min_loss 初始化为正无穷
    start_time = time.time()  ## 计时
    optimizer = optim.AdamW(model.parameters(), 5e-4)
    start_epoch = (load_checkpoint(ckpt_save_path, model, optimizer)
                   if args.is_load_ckpt else 1)  ## 如果load ckpt则从导入的epoch开始继续训练，否则从1开始
    # lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)  ## 学习率指数衰减
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,  ## loss在一定次数内不再下降则衰减学习率
                                                        mode='min',  ## 记录最小值，loss大于最小值则计数一次
                                                        factor=0.9,  ## 衰减因子
                                                        patience=400,  ## 计数值
                                                        verbose=False)  ## 是否将lr变化打印出来

    animator = Animator(xlabel='epoch', ylabel='loss', xlim=[1, args.iter + 1], ylim=[10.690, 10.900],
                        legend='train_loss')  ## 添加loss曲线图参数
    for i in range(start_epoch, args.iter):
        optimizer.zero_grad()
        log_px = model.log_prob(train_data)
        loss = -log_px.mean()
        loss.backward()
        optimizer.step()
        # lr_scheduler.step(loss)
        if i % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print("iter: {}, loss: {:.4f}, lr: {:.6f}".format(i, loss.detach().cpu().numpy(), current_lr))
            animator.add(i, [loss.detach().cpu().numpy()])
        if i % 25000 == 0:  ## 每过25000次训练步则保存一次模型，优化器等参数
            save_checkpoint(model, optimizer, i, loss, ckpt_save_path)
        if i > 25000:  ## 25000步之后，开始记录训练过程中最小的loss对应的模型参数而非训练结束后的模型参数
            if loss < min_loss:
                min_loss = loss
                torch.save(model.state_dict(), save_path)
    train_time = (time.time() - start_time) / 60  ## 计时 分钟
    print(f"total training time: {train_time:.2f} minutes")
    print(f"the min loss: {min_loss:.2f}")
    # torch.save(model.state_dict(), save_path)
    plt.savefig(loss_save_path, bbox_inches='tight')
    plt.show()


def inference(model, save_path, train_data, images_save_path, device):
    model.load_state_dict(torch.load(save_path))  ## 导入训练结束时保存的参数
    model.eval()  ## 模型评估模式，会改变BN层或者dropout层的参数
    with torch.no_grad():
        sample = model.sample((100000,))
        sample = sample.cpu().numpy()
        train_data = train_data.numpy()

        x = np.linspace(0, 300, 100)
        y = np.linspace(0, 300, 100)
        points = np.vstack(np.meshgrid(x, y)).reshape([2, -1]).T  ## 得到100*100的网格点

        z_t1 = torch.tensor(points).type(torch.float32).to(device)
        z_density = torch.exp(model.log_prob(z_t1))  ## 得到网格点处，p(x)的值， p为训练数据的PDF
        print(f"the integral of density: {(sum(z_density)):.2f}")

        fig = plt.figure(figsize=(18, 6), dpi=200)  ## dpi是图像清晰度
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_title('Target')
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_title('Sample')
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_title('Density')
        ax1.hist2d(train_data[:, 0], train_data[:, 1], bins=300, density=True, cmap='plasma')
        ax2.hist2d(sample[:, 0], sample[:, 1], bins=300, density=True, cmap='plasma')
        ax3.contourf(x, y, z_density.cpu().numpy().reshape(100, 100), levels=50, origin='lower', cmap='plasma')
        plt.savefig(images_save_path, bbox_inches='tight')
        plt.show()
        # plt.scatter(sample[:, 0], sample[:, 1])
        # plt.show()


def load_data(file_path):  ## 院徽数据
    data = np.loadtxt(file_path)
    data = torch.from_numpy(data).float()
    return data


def main(args):
    if args.data_from == "original":
        train_data = original_dist(512)
    elif args.data_from == "my_data":
        train_data = load_data(args.data_dir)
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')
    prior_dist = DiagGaussian(torch.tensor([0.0, 0.0], device=device),
                              torch.tensor([[1., 0.0], [0.0, 1.]], device=device))
    model = KRnet(prior_dist, input_size=args.input_size, n_step=args.n_step, n_depth=args.n_depth, width=args.width,
                  n_hidden=args.n_hidden).to(device)
    images_save_path = os.path.join(args.images_dir,
                                    "image_minloss_iter{}_{}_{}_{}_{}.png".format(args.iter, args.data_from,
                                                                                  args.n_depth,
                                                                                  args.width, args.n_hidden))
    loss_save_path = os.path.join(args.loss_dir,
                                  "loss_minloss_{}_{}_{}_{}.png".format(args.data_from,
                                                                        args.n_depth,
                                                                        args.width, args.n_hidden))
    param_save_path = os.path.join(args.param_path,
                                   "model_minloss_iter{}_{}_{}_{}_{}.pth".format(args.iter, args.data_from,
                                                                                 args.n_depth,
                                                                                 args.width, args.n_hidden))
    ckpt_save_path = os.path.join(args.ckpt_dir, "ckpt_iter{}.pth".format(args.iter))
    if args.mode == "train":
        train_model(model, param_save_path, train_data.to(device), loss_save_path, ckpt_save_path)
    elif args.mode == "test":
        inference(model, param_save_path, train_data, images_save_path, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=5000)
    parser.add_argument("--data_dir", default="./data/point_20000_new.txt")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mode", default="train")
    parser.add_argument("--param_path", default="param")
    parser.add_argument("--input_size", type=int, default=2)  ## 数据维度
    parser.add_argument("--n_step", type=int, default=1)  ## KRnet框架下数据每次衰减的维度
    parser.add_argument("--n_depth", type=int, default=10)  ## 仿射耦合层堆积层数
    parser.add_argument("--width", type=int, default=32)  ## 仿射耦合层中隐藏层神经元数
    parser.add_argument("--n_hidden", type=int, default=5)  ## 仿射耦合层中隐藏层个数
    parser.add_argument("--data_from", choices=["original", "my_data"], default="my_data")
    parser.add_argument("--images_dir", default="./images")
    parser.add_argument("--loss_dir", default="./loss_curve")
    parser.add_argument("--ckpt_dir", default="./ckpt")
    parser.add_argument('--is_load_ckpt', action='store_true',  ## 是否导入ckpt继续训练，否则重新训练
                        help='Load model from checkpoint if specified')  ##store_true 即在命令行输入--is_load_ckpt 即为True
    args = parser.parse_args()
    ## 命令行运行： python main.py --mode train 即可训练
    main(args)
