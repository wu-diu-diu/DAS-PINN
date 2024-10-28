import os.path

from layers import *
from distributions import DiagGaussian, original_dist
import torch.optim as optim
import argparse
from utils import Animator
import matplotlib.pyplot as plt
import time


def train_model(model, save_path, train_data, loss_save_path):
    start_time = time.time()
    optimizer = optim.AdamW(model.parameters(), 5e-4)
    # lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode='min',
                                                        factor=0.99,
                                                        patience=400,
                                                        verbose=True)
    animator = Animator(xlabel='epoch', ylabel='loss', xlim=[1, args.iter + 1], ylim=[10.690, 10.900],
                        legend='train_loss')
    for i in range(1, args.iter):
        optimizer.zero_grad()
        log_px = model.log_prob(train_data)
        loss = -log_px.mean()
        loss.backward()
        optimizer.step()
        lr_scheduler.step(loss)
        if i % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print("iter: {}, loss: {:.4f}, lr: {:.6f}".format(i, loss.detach().cpu().numpy(), current_lr))
            animator.add(i, [loss.detach().cpu().numpy()])
    train_time = (time.time() - start_time) / 60
    print(f"total training time: {train_time:.2f} minutes")
    torch.save(model.state_dict(), save_path)
    plt.savefig(loss_save_path, bbox_inches='tight')
    plt.show()


def inference(model, save_path, train_data, images_save_path):
    model.load_state_dict(torch.load(save_path))
    model.eval()
    with torch.no_grad():
        sample = model.sample((100000,))
        sample = sample.cpu().numpy()
        train_data = train_data.numpy()

        fig = plt.figure(figsize=(12, 6), dpi=200)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title('Target')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title('Sample')
        ax1.hist2d(train_data[:, 0], train_data[:, 1], bins=300, density=True, cmap='plasma')
        ax2.hist2d(sample[:, 0], sample[:, 1], bins=300, density=True, cmap='plasma')
        plt.savefig(images_save_path, bbox_inches='tight')
        plt.show()
        # plt.scatter(sample[:, 0], sample[:, 1])
        # plt.show()


def load_data(file_path):
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
                                    "image_lr_shrink_iter{}_{}_{}_{}_{}.png".format(args.iter, args.data_from,
                                                                                    args.n_depth,
                                                                                    args.width, args.n_hidden))
    loss_save_path = os.path.join(args.loss_dir,
                                  "loss_lr_shrink_{}_{}_{}_{}.png".format(args.data_from,
                                                                          args.n_depth,
                                                                          args.width, args.n_hidden))
    param_save_path = os.path.join(args.param_path,
                                   "model_lr_shrink_iter{}_{}_{}_{}_{}.pth".format(args.iter, args.data_from,
                                                                                   args.n_depth,
                                                                                   args.width, args.n_hidden))
    if args.mode == "train":
        train_model(model, param_save_path, train_data.to(device), loss_save_path)
    elif args.mode == "test":
        inference(model, param_save_path, train_data, images_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=5000)
    parser.add_argument("--data_dir", default="./data/point_20000_new.txt")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mode", default="train")
    parser.add_argument("--param_path", default="param")
    parser.add_argument("--input_size", type=int, default=2)
    parser.add_argument("--n_step", type=int, default=1)
    parser.add_argument("--n_depth", type=int, default=10)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--n_hidden", type=int, default=5)
    parser.add_argument("--data_from", choices=["original", "my_data"])
    parser.add_argument("--images_dir", default="./images")
    parser.add_argument("--loss_dir", default="./loss_curve")
    args = parser.parse_args()
    main(args)
