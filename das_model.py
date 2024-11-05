import os

import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.interpolate import griddata

from layers import KRnet
import torch.optim as optim
from utils import My_MLP, projection_onto_infunitball, get_train_data, get_valid_data, exact_solution
from equation import Helmholtz
from torch.utils.data import TensorDataset, DataLoader
from distributions import DiagGaussian
import time


class DAS():
    def __init__(self, args, device, prior_dist):
        self.args = args
        self.device = device
        self.pdf_model = KRnet(prior_dist, self.args.input_size, self.args.flow_n_step, self.args.flow_n_depth,
                               self.args.flow_width, self.args.flow_n_hidden,
                               device=self.device).to(self.device)
        self.net = My_MLP(self.args.input_size, self.args.pde_n_hidden, self.args.pde_width, self.args.pde_output_size,
                          self.args.pde_activation).to(self.device)
        self.pde_optimizer = optim.AdamW(self.net.parameters(), 1e-4)
        self.flow_optimizer = optim.AdamW(self.pdf_model.parameters(), 1e-4)

    def get_pde_loss(self, x, stage_idx):
        res, u_pred_train = Helmholtz(x, self.net)
        pde_loss = torch.mean(res ** 2)
        return pde_loss, res, u_pred_train

    def get_pdf(self, x):
        log_pdfx = self.pdf_model.log_prob(x)
        pdfx = torch.exp(log_pdfx)
        return pdfx

    def get_entropy_loss(self, quantity, pre_pdf, x, i):
        """
        quantity: (num,1)
        pre,pdf:(num, 1)
        x:(num, dim)
        """
        log_pdf = self.pdf_model.log_prob(x)
        log_pdf = torch.clamp(log_pdf, min=-23.02585, max=5.0)

        # scaling for numerical stability
        scaling = 1000.0
        pre_pdf = scaling * pre_pdf
        quantity = scaling * quantity

        # importance sampling
        ratio = quantity / pre_pdf
        res_time_logpdf = ratio * log_pdf
        entropy_loss = -torch.mean(res_time_logpdf)
        return entropy_loss

    def resample(self):
        n_resample = self.args.num_sample
        projection_operator = projection_onto_infunitball
        x_candidate = self.pdf_model.sample((n_resample,))
        x_resample, x_bd = projection_operator(x_candidate.detach().cpu().numpy())
        nv_sample = x_resample.shape[0]
        while nv_sample < n_resample:
            n_diff = n_resample - nv_sample
            # x_prior_new = self.pdf_model.draw_samples_from_prior(n_diff, args.n_dim)
            # x_candidate_new = self.pdf_model.mapping_from_prior(x_prior_new).numpy()
            x_candidate_new = self.pdf_model.sample((n_diff,))
            x_candidate_new, _ = projection_operator(x_candidate_new.detach().cpu().numpy())
            x_resample = np.concatenate((x_resample, x_candidate_new), axis=0)
            # x_bd = np.concatenate((x_bd, x_bd_new), axis=0)
            nv_sample = x_resample.shape[0]

        if x_bd.shape[0] < x_resample.shape[0]:
            n_add = x_resample.shape[0] - x_bd.shape[0]
            _, x_bd_add = get_train_data(2, n_add, n_add, 1, device=self.device)
            x_bd_add = x_bd_add.detach().cpu().numpy()
            x_bd = np.concatenate((x_bd, x_bd_add), axis=0)
        else:
            n_s = x_resample.shape[0]
            x_bd = x_bd[:n_s, :]

        # return x_add
        # x_new = np.concatenate((x_resample, x_bd), axis=0)
        # x_new = torch.tensor(x_new, dtype=torch.float32, requires_grad=True).to(device)
        x_new = torch.tensor(x_resample, dtype=torch.float32, requires_grad=True).to(self.device)
        return x_new

    def solve_pde(self, train_data, stage_idx, valid_data):
        n_epochs = self.args.pde_epoch
        for k in range(1, n_epochs + 1):
            self.pde_optimizer.zero_grad()
            pde_loss, res_train, u_pred_train = self.get_pde_loss(train_data, stage_idx)
            pde_loss.backward()
            self.pde_optimizer.step()
            res_loss = torch.mean(res_train)
            print(f"stage:{stage_idx}, epoch:{k}, res_loss:{res_loss.detach().cpu().numpy():.4f}, "
                  f"pde_loss:{pde_loss.detach().cpu().numpy():.4f}")
        with torch.no_grad():
            u_pred_valid = self.net(valid_data)
        return u_pred_valid, res_train, u_pred_train

    def solve_flow(self, train_data, stage_idx):
        flow_epochs = self.args.flow_epoch
        for k in range(1, flow_epochs + 1):
            self.flow_optimizer.zero_grad()
            _, quantity, _ = self.get_pde_loss(train_data, stage_idx)  ## 得到采样点关于pde的残差
            quantity = torch.abs(quantity)  ## 令残差大的地方，P（x）更大，无关正负
            pre_pdf = torch.ones_like(quantity, dtype=torch.float32).to(self.device)  ## 重要性采样
            entropy_loss = self.get_entropy_loss(quantity, pre_pdf, train_data, stage_idx)
            entropy_loss.backward()
            self.flow_optimizer.step()
            quantity_loss = torch.mean(quantity)
            print(
                f"stage:{stage_idx}, flow_epoch:{k}, quantity:{quantity_loss.detach().cpu().numpy():.4f}, entropy:{entropy_loss.detach().cpu().numpy():.4f}")

    def train(self, train_data, valid_data):
        max_stage = self.args.stage
        print('====== Training process starting... ======')
        flow_data = train_data
        x = train_data
        for i in range(1, max_stage + 1):

            u_pred_valid, res_train, u_pred_train = self.solve_pde(train_data, i, valid_data)
            u_true_valid = exact_solution(valid_data)

            self.plot_pred(u_true_valid, u_pred_valid, i)  ## 绘制
            self.plot_res(train_data, res_train, i)  ## 绘制训练数据在pde上的残差
            if i < max_stage:

                self.solve_flow(flow_data, i)
                with torch.no_grad():
                    x_new = self.resample()
                self.plot_sample(x_new, i)
                num_sample = x_new.shape[0]
                # train_data = x_new
                train_data = torch.concat([x, x_new], dim=0)  ## 每次训练后的点都加载原始随机点上
                # flow_data = self.pdf_model.sample((num_sample,))
                # flow_data = flow_data.clone().detach().float().requires_grad_(True)

            # image_save_path = os.path.join(self.args.images_dir, "das_res_stage{}.pth".format(i))
            # param_pde_model = os.path.join(self.args.param_path, "das_pde_stage{}.pth".format(i))
            # torch.save(self.net.state_dict(), param_pde_model)
            # param_flow_model = os.path.join(self.args.param_path, "das_flow_stage{}.pth".format(i))
            # torch.save(self.pdf_model.state_dict(), param_flow_model)

    # def inference(self):
    #     self.net.eval()
    #     with torch.no_grad():


    def plot_res(self, train_data, res, i):
        x = train_data[:, 0].detach().cpu().numpy()
        y = train_data[:, 1].detach().cpu().numpy()
        values = res.detach().cpu().numpy()
        grid_resolution = 200  # 可以设置为更高的值以增加精度
        grid_x, grid_y = np.mgrid[min(x):max(x):grid_resolution * 1j, min(y):max(y):grid_resolution * 1j]
        grid_z = griddata((x, y), values, (grid_x, grid_y), method='linear')
        grid_z = np.abs(grid_z)

        plt.figure(figsize=(10, 8))
        plt.pcolormesh(grid_x, grid_y, grid_z, shading='auto', cmap='viridis')  # 使用 shading='auto' 处理边缘问题
        plt.colorbar(label='Values')  # 添加颜色条
        plt.title('Pcolormesh of Values')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.savefig(os.path.join(self.args.images_dir, 'res_{}.png'.format(i)), dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()

    def plot_sample(self, sample, i):
        plt.scatter(sample[:, 0].detach().cpu().numpy(), sample[:, 1].detach().cpu().numpy(), s=0.5)
        plt.savefig(os.path.join(self.args.images_dir, 'sample_{}.png'.format(i)))
        # plt.show()
        plt.close()

    def plot_pred(self, u_true, u_pred, i):
        x_test = np.linspace(-1, 1, 256)
        y_test = np.linspace(-1, 1, 256)
        X, Y = np.meshgrid(x_test, y_test)
        u_true = u_true.cpu().numpy().reshape(X.shape)
        u_pred = u_pred.cpu().numpy().reshape(X.shape)
        res = np.abs(u_pred - u_true)

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        pc_exact = axs[0].pcolormesh(X, Y, u_true, shading='auto', cmap='viridis')
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
        plt.savefig(os.path.join(self.args.images_dir, 'pred_{}.png'.format(i)))
        plt.tight_layout()
        # plt.show()
        plt.close()

# start_time = time.time()
# test = DAS()
# test.train()
# end_time = (time.time() - start_time) / 60
# print(f"total training time: {end_time:.2f} minutes")
