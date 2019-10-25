import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(torch.cat([init_dim * torch.arange(n_tile) + i for i in range(init_dim)])).cuda()
    return torch.index_select(a, dim, order_index)


class Conv2d_STN(nn.Module):
    def __init__(self, *argv, **argn):
        super(Conv2d_STN, self).__init__()
        self.stn0 = nn.Conv2d(in_channels=argv[0], out_channels=argv[0], kernel_size=argv[2], padding=argv[2] / 2, **argn)
        self.stn1 = nn.Conv2d(in_channels=argv[0], out_channels=6, kernel_size=argv[2], bias=False)
        #self.stn0 = nn.Conv2d(in_channels=argv[0], out_channels=argv[0], kernel_size=argv[2])
        #self.stn1 = nn.Conv2d(in_channels=argv[0], out_channels=6, kernel_size=1, bias=False)
        self.conv = nn.Linear(argv[0] * argv[2] * argv[2], argv[1], **argn)
        self.stn1.weight.data.zero_()
        self.theta_mask = torch.tensor([1, 1, 0, 1, 1, 0])
        self.identity = torch.tensor([1.0, 0., 0., 0., 1., 0.]).cuda().view(1, 2, 3)
        self.theta = []


    def compute_scores(self, x, check):
        out = self.stn1(F.relu(self.stn0(x))) * (1 - check) + torch.tensor([1., 0., 0., 0., 1., 0.]).cuda().view(1, 6, 1, 1)
        #temp = [out[:, i] * self.theta_mask[i] for i in range(6)]
        #out = torch.cat(temp, 1).view(out.shape)
        theta = out.permute(0, 3, 2, 1).contiguous().view(out.shape[0] * out.shape[2] * out.shape[3], 2, 3)
        grid = F.affine_grid(theta, (theta.shape[0], x.shape[1], self.stn1.kernel_size[0], self.stn1.kernel_size[1]))
        xv, yv = torch.meshgrid(
            [torch.arange(self.stn1.kernel_size[0] / 2, self.stn1.kernel_size[0] / 2 + out.shape[2]),
             torch.arange(self.stn1.kernel_size[0] / 2, self.stn1.kernel_size[0] / 2 + out.shape[3])])
        auxx = ((grid[:, :, :, 0] * (self.stn1.kernel_size[0] / 2)).view(out.shape[0], out.shape[2], out.shape[3],
                self.stn1.kernel_size[0], self.stn1.kernel_size[1]) + xv.view(1, out.shape[2], out.shape[3], 1, 1).cuda().float()).view(
            out.shape[0] * out.shape[2] * out.shape[3], self.stn1.kernel_size[0], self.stn1.kernel_size[1]) / (x.shape[2] - 1) * 2 - 1
        auxy = ((grid[:, :, :, 1] * (self.stn1.kernel_size[1] / 2)).view(out.shape[0], out.shape[2], out.shape[3],
                self.stn1.kernel_size[0], self.stn1.kernel_size[1]) + yv.view(1, out.shape[2], out.shape[3], 1, 1).cuda().float()).view(
            out.shape[0] * out.shape[2] * out.shape[3], self.stn1.kernel_size[0], self.stn1.kernel_size[1]) / (x.shape[3] - 1) * 2 - 1
        auxgrid = torch.stack((auxx, auxy), 3)
        xs = F.grid_sample(tile(x, 0, out.shape[2] * out.shape[3]), auxgrid)
        x_out = self.conv(xs.view(xs.shape[0], -1))
        output = x_out.view(out.shape[0], out.shape[2], out.shape[3], x_out.shape[1]).permute(0, 3, 2, 1)
        self.theta = theta
        return output, theta, out, auxx, auxy


    def forward(self, x, check=0):
        score, theta, out, auxx, auxy = self.compute_scores(x, check)
        return score, (self.identity-theta), out, auxx, auxy

