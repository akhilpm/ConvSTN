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
        self.num_classes = argv[3]
        self.stn0 = nn.Conv2d(in_channels=argv[0], out_channels=argv[0], kernel_size=argv[2], padding=argv[2] / 2, **argn)
        self.stn1 = nn.Conv2d(in_channels=argv[0], out_channels=6, kernel_size=argv[2], bias=False)
        #self.stn0 = nn.Conv2d(in_channels=argv[0], out_channels=argv[0], kernel_size=argv[2])
        #self.stn1 = nn.Conv2d(in_channels=argv[0], out_channels=6, kernel_size=1, bias=False)
        self.conv = nn.Linear(argv[0] * argv[2] * argv[2], argv[1], **argn)
        #self.conv_eq = nn.Conv2d(*argv, **argn)
        #self.conv_eq.weight = torch.nn.Parameter(self.conv.weight.view(self.conv_eq.weight.shape))
        #self.conv_eq.bias = torch.nn.Parameter(self.conv.bias.view(self.conv_eq.bias.shape))
        self.stn1.weight.data.zero_()
        self.theta_mask = torch.tensor([1, 1, 0, 1, 1, 0])
        self.conv_last = nn.Conv2d(argv[1], self.num_classes, kernel_size=1)
        self.identity = torch.tensor([1.0, 0., 0., 0., 1., 0.]).cuda().view(1, 2, 3)
        #self.identity = torch.tensor([0.25, 0., 0., 0., 0.25, 0.]).cuda().view(1, 2, 3)
        #self.stn1.bias.data.zero_()
        self.theta = []
        #self.filtersize = self.stn1.kernel_size

    def get_no_transform_box(self, theta, x, out, detection_locs, image_dim, conf):
        no_transform = torch.tensor([1., 0., 0., 0., 1., 0.]).cuda().view(1,2,3)
        theta_NT = torch.zeros(theta.shape).cuda().copy_(no_transform)
        grid = F.affine_grid(theta_NT, (theta.shape[0], x.shape[1], self.stn1.kernel_size[0], self.stn1.kernel_size[1]))
        xv, yv = torch.meshgrid(
        [torch.arange(self.stn1.kernel_size[0] / 2, self.stn1.kernel_size[0] / 2 + out.shape[2]),
         torch.arange(self.stn1.kernel_size[0] / 2, self.stn1.kernel_size[0] / 2 + out.shape[3])])
        auxx = ((grid[:, :, :, 0] * (self.stn1.kernel_size[0] / 2)).view(out.shape[0], out.shape[2], out.shape[3],
                 self.stn1.kernel_size[0], self.stn1.kernel_size[1]) + xv.view(1, out.shape[2], out.shape[3], 1, 1).cuda().float()).view(
            out.shape[0] * out.shape[2] * out.shape[3], self.stn1.kernel_size[0], self.stn1.kernel_size[1]) / (x.shape[2] - 1) * 2 - 1
        auxy = ((grid[:, :, :, 1] * (self.stn1.kernel_size[1] / 2)).view(out.shape[0], out.shape[2], out.shape[3],
             self.stn1.kernel_size[0], self.stn1.kernel_size[1]) + yv.view(1, out.shape[2], out.shape[3], 1, 1).cuda().float()).view(
            out.shape[0] * out.shape[2] * out.shape[3], self.stn1.kernel_size[0], self.stn1.kernel_size[1]) / (x.shape[3] - 1) * 2 - 1

        tempx, tempy = (auxx + 1) * 0.5, (auxy + 1) * 0.5
        tempx, tempy = tempx.view(theta.shape[0], -1), tempy.view(theta.shape[0], -1)
        min_x, min_y = (torch.min(tempx[detection_locs], dim=1)[0] * image_dim).clamp(0, image_dim), (
                    torch.min(tempy[detection_locs], dim=1)[0] * image_dim).clamp(0, image_dim)
        max_x, max_y = (torch.max(tempx[detection_locs], dim=1)[0] * image_dim).clamp(0, image_dim), (
                    torch.max(tempy[detection_locs], dim=1)[0] * image_dim).clamp(0, image_dim)
        box = torch.stack([min_x, min_y, max_x, max_y, conf], dim=1)
        return box


    def forward(self, x, image_dim, check=0):
        batch_size = x.shape[0]
        out = self.stn1(F.relu(self.stn0(x))) * (1 - check) + torch.tensor([1., 0., 0., 0., 1., 0.]).cuda().view(1, 6, 1, 1)
        #temp = [out[:, i] * self.theta_mask[i] for i in range(6)]
        #out = torch.cat(temp, 1).view(out.shape)
        theta = out.permute(0, 3, 2, 1).contiguous().view(out.shape[0] * out.shape[2] * out.shape[3], 2, 3)
        grid = F.affine_grid(theta, (theta.shape[0], x.shape[1], self.stn0.kernel_size[0], self.stn0.kernel_size[1]))
        xv, yv = torch.meshgrid(
            [torch.arange(self.stn0.kernel_size[0] / 2, self.stn0.kernel_size[0] / 2 + out.shape[2]),
             torch.arange(self.stn0.kernel_size[0] / 2, self.stn0.kernel_size[0] / 2 + out.shape[3])])
        auxx = ((grid[:, :, :, 0] * (self.stn0.kernel_size[0] / 2)).view(out.shape[0], out.shape[2], out.shape[3],
                self.stn0.kernel_size[0], self.stn0.kernel_size[1]) + xv.view(1, out.shape[2], out.shape[3], 1, 1).cuda().float()).view(
            out.shape[0] * out.shape[2] * out.shape[3], self.stn0.kernel_size[0], self.stn0.kernel_size[1]) / (x.shape[2] - 1) * 2 - 1
        auxy = ((grid[:, :, :, 1] * (self.stn0.kernel_size[1] / 2)).view(out.shape[0], out.shape[2], out.shape[3],
                self.stn0.kernel_size[0], self.stn0.kernel_size[1]) + yv.view(1, out.shape[2], out.shape[3], 1, 1).cuda().float()).view(
            out.shape[0] * out.shape[2] * out.shape[3], self.stn0.kernel_size[0], self.stn0.kernel_size[1]) / (x.shape[3] - 1) * 2 - 1
        auxgrid = torch.stack((auxx, auxy), 3)
        xs = F.grid_sample(tile(x, 0, out.shape[2] * out.shape[3]), auxgrid)
        x_out = self.conv(xs.view(xs.shape[0], -1))
        output = x_out.view(out.shape[0], out.shape[2], out.shape[3], x_out.shape[1]).permute(0, 3, 2, 1)
        output = self.conv_last(F.relu(output))
        self.theta = theta
        #print(auxx.shape)
        #if check:
        #    output_conv = self.conv_eq(x)
        #    print(torch.mean(output-output_conv))

        output = output.contiguous().view(out.shape[0], out.shape[2]*out.shape[3]*self.num_classes)
        output = F.softmax(output, dim=1)
        output = output.view(out.shape[0], self.num_classes, out.shape[2], out.shape[3])
        likelihood = output.view(batch_size, self.num_classes, -1).sum(2)
        #bboxes = {k: np.zeros((out.shape[2]*out.shape[3], 5)) for k in range(batch_size)}

        with torch.no_grad():
            pred = torch.argmax(likelihood, dim=1)
            scores = output[torch.arange(batch_size), pred].view(batch_size, -1)
            conf, detection_locs = torch.max(scores, dim=1)
            loc_y, loc_x = detection_locs/out.shape[3], detection_locs%out.shape[3]
            detection_locs = loc_x*out.shape[2] + loc_y
            #print(scores[0], output[0, pred[0]], detection_locs[0])
            start = torch.arange(batch_size).cuda() * (out.shape[2]*out.shape[3])
            detection_locs = detection_locs + start
            #tempx, tempy = (auxx+1) * 0.5, (auxy+1) * 0.5
            tempx, tempy = (auxx + 1) * 0.5, (auxy + 1) * 0.5
            tempx, tempy = tempx.view(theta.shape[0], -1), tempy.view(theta.shape[0], -1)
            #u,v = tempx[0], tempy[0]
            #min_x, min_y = (torch.min(tempx[detection_locs], dim=1)[0] * image_dim).clamp(0, image_dim-1), (torch.min(tempy[detection_locs], dim=1)[0] * image_dim).clamp(0, image_dim-1)
            #max_x, max_y = (torch.max(tempx[detection_locs], dim=1)[0] * image_dim).clamp(0, image_dim-1), (torch.max(tempy[detection_locs], dim=1)[0] * image_dim).clamp(0, image_dim-1)
            min_x, min_y = (torch.min(tempx[detection_locs], dim=1)[0] * image_dim).clamp(0, image_dim), (torch.min(tempy[detection_locs], dim=1)[0] * image_dim).clamp(0, image_dim)
            max_x, max_y = (torch.max(tempx[detection_locs], dim=1)[0] * image_dim).clamp(0, image_dim), (torch.max(tempy[detection_locs], dim=1)[0] * image_dim).clamp(0, image_dim)
            box = torch.stack([min_x, min_y, max_x, max_y, conf], dim=1)
            box_NT = self.get_no_transform_box(theta, x, out, detection_locs, image_dim, conf)
            #win = torch.max(likelihood, dim=1)[0]
            #print(win)
            #print(box, box_NT)

            #min_x, min_y = (torch.min(tempx, dim=1)[0] * image_dim).clamp(0, image_dim-1), (torch.min(tempy, dim=1)[0] * image_dim).clamp(0, image_dim-1)
            #max_x, max_y = (torch.max(tempx, dim=1)[0] * image_dim).clamp(0, image_dim-1), (torch.max(tempy, dim=1)[0] * image_dim).clamp(0, image_dim-1)
            #min_x, min_y = torch.split(min_x, out.shape[2]*out.shape[3]), torch.split(min_y, out.shape[2]*out.shape[3])
            #max_x, max_y = torch.split(max_x, out.shape[2]*out.shape[3]), torch.split(max_y, out.shape[2]*out.shape[3])
            #for i in range(batch_size):
            #    conf = output[i, pred[i]].t().contiguous().view(-1)
            #    bboxes[i] = torch.stack([min_x[i], min_y[i], max_x[i], max_y[i], conf], dim=1).cpu().numpy()
        return likelihood, output, box, box_NT, (self.identity - theta)
