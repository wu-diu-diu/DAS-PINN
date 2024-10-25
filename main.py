import numpy as np
from layers import *
from distributions import DiagGaussian
import torch.optim as optim

if __name__ == '__main__':

    file_path = './data/point_final.txt'
    point = np.loadtxt(file_path)
    tensor_point = torch.from_numpy(point).float()

    device = torch.device('cuda:0')
    Pz = DiagGaussian(torch.tensor([0.0, 0.0]), torch.tensor([[1., 0.0], [0.0, 1.]]))
    flow = KRnet(Pz, input_size=2, n_step=1, n_depth=8, width=24, n_hidden=2, device=device)
    optimizer = optim.AdamW(flow.parameters(), 1e-5)
    for i in range(1, 5001):
        optimizer.zero_grad()
        log_px = flow.log_prob(tensor_point)
        loss = -log_px.mean()
        loss.backward()
        optimizer.step()
        print(loss)



