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
        self.num_classes = 201
        self.stn0 = nn.Conv2d(in_channels=argv[0], out_channels=argv[0], kernel_size=argv[2], padding=argv[2] / 2, **argn)
        self.stn1 = nn.Conv2d(in_channels=argv[0], out_channels=6, kernel_size=argv[2], bias=False)
        self.conv = nn.Linear(argv[0] * argv[2] * argv[2], argv[1], **argn)
        self.stn1.weight.data.zero_()
        self.theta_mask = torch.tensor([1, 1, 0, 1, 1, 0])
        #self.conv_last_l5 = nn.Conv2d(argv[1], self.num_classes, kernel_size=1)
        self.conv_last = nn.Conv2d(argv[1], self.num_classes, kernel_size=1)
        self.identity = torch.tensor([1.0, 0., 0., 0., 1., 0.]).cuda().view(1, 2, 3)
        self.theta = []

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
        min_x, min_y = (torch.min(tempx[detection_locs], dim=1)[0] * image_dim).clamp(0, image_dim - 1), (
                    torch.min(tempy[detection_locs], dim=1)[0] * image_dim).clamp(0, image_dim - 1)
        max_x, max_y = (torch.max(tempx[detection_locs], dim=1)[0] * image_dim).clamp(0, image_dim - 1), (
                    torch.max(tempy[detection_locs], dim=1)[0] * image_dim).clamp(0, image_dim - 1)
        box = torch.stack([min_x, min_y, max_x, max_y, conf], dim=1)
        return box

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
        output = self.conv_last(F.relu(output))
        self.theta = theta
        return output, theta, out, auxx, auxy

    def get_max_boxes(self, output, pred, out, auxx, auxy, image_dim):
        batch_size = output.shape[0]
        n_thetas = out.shape[0] * out.shape[2] * out.shape[3]
        scores = output[torch.arange(batch_size), pred]
        conf, detection_locs = torch.max(scores, dim=1)
        loc_y, loc_x = detection_locs / out.shape[2], detection_locs % out.shape[2]
        detection_locs = loc_x * out.shape[2] + loc_y
        start = torch.arange(batch_size).cuda() * (out.shape[2] * out.shape[3])
        detection_locs = detection_locs + start

        tempx, tempy = (auxx + 1) * 0.5, (auxy + 1) * 0.5
        tempx, tempy = tempx.view(n_thetas, -1), tempy.view(n_thetas, -1)
        min_x, min_y = (torch.min(tempx[detection_locs], dim=1)[0] * image_dim).clamp(0, image_dim - 1), (
                torch.min(tempy[detection_locs], dim=1)[0] * image_dim).clamp(0, image_dim - 1)
        max_x, max_y = (torch.max(tempx[detection_locs], dim=1)[0] * image_dim).clamp(0, image_dim - 1), (
                torch.max(tempy[detection_locs], dim=1)[0] * image_dim).clamp(0, image_dim - 1)
        box = torch.stack([min_x, min_y, max_x, max_y, conf], dim=1)
        return box, detection_locs, conf

    def forward(self, x, log, check=0, image_dim=320):
        batch_size = x[0].shape[0]
        score_p5, theta_l5, out_l5, auxx_l5, auxy_l5 = self.compute_scores(x[0], check)
        #score_p5 = self.conv_last1(F.relu(score_p5))
        score_p4, theta_l4, out_l4, auxx_l4, auxy_l4 = self.compute_scores(x[1], check)
        #score_p4 = self.conv_last2(F.relu(score_p4))
        #score_p3, theta_l3, out_l3, auxx_l3, auxy_l3 = self.compute_scores(x[2], check)
        theta_diff = torch.cat([(self.identity-theta_l5), (self.identity-theta_l4)], dim=0)
        #theta_diff = torch.cat([(self.identity - theta_l5), (self.identity - theta_l4), (self.identity - theta_l3)], dim=0)
        score_p5 = score_p5.view(batch_size, self.num_classes, -1)
        score_p4 = score_p4.view(batch_size, self.num_classes, -1)
        #score_p3 = score_p3.view(batch_size, self.num_classes, -1)
        score = torch.cat([score_p5, score_p4], dim=2)
        #score = torch.cat([score_p5, score_p4, score_p3], dim=2)

        joint = F.softmax(score.view(batch_size, -1), dim=1).view(batch_size, self.num_classes, -1)
        likelihood =  joint.sum(2)
        l5 = (joint[:, :, :score_p5.shape[2]]).contiguous().view(batch_size, self.num_classes, -1)
        l4 = (joint[:, :, score_p5.shape[2]:score_p5.shape[2]+score_p4.shape[2]]).contiguous().view(batch_size, self.num_classes, -1)
        #l3 = (joint[:, :, score_p5.shape[2]+score_p4.shape[2]:]).contiguous().view(batch_size, self.num_classes, -1)

        with torch.no_grad():
            pred = torch.argmax(likelihood, dim=1)
            box_l5, detection_locs, conf = self.get_max_boxes(l5, pred, out_l5, auxx_l5, auxy_l5, image_dim)
            box_NT_l5 = self.get_no_transform_box(theta_l5, x[0], out_l5, detection_locs, image_dim, conf)
            box_l4, detection_locs, conf = self.get_max_boxes(l4, pred, out_l4, auxx_l4, auxy_l4, image_dim)
            box_NT_l4 = self.get_no_transform_box(theta_l4, x[1], out_l4, detection_locs, image_dim, conf)
            #box_l3, detection_locs, conf = self.get_max_boxes(l3, pred, out_l3, auxx_l3, auxy_l3, image_dim)
            #box_NT_l3 = self.get_no_transform_box(theta_l3, x[2], out_l3, detection_locs, image_dim, conf)
            boxes = torch.cat([torch.unsqueeze(box_l5, dim=1), torch.unsqueeze(box_l4, dim=1)], dim=1)
            boxes_NT = torch.cat([torch.unsqueeze(box_NT_l5, dim=1), torch.unsqueeze(box_NT_l4, dim=1)], dim=1)
            #boxes = torch.cat([torch.unsqueeze(box_l5, dim=1), torch.unsqueeze(box_l4, dim=1), torch.unsqueeze(box_l3, dim=1)], dim=1)
            #boxes_NT = torch.cat([torch.unsqueeze(box_NT_l5, dim=1), torch.unsqueeze(box_NT_l4, dim=1), torch.unsqueeze(box_NT_l3, dim=1)], dim=1)

            #max_of_all_likelihoods = torch.stack([box_l5[:, 4], box_l4[:, 4]], dim=1)
            #max_of_all_likelihoods = torch.stack([box_l5[:, 4], box_l4[:, 4]], dim=1)
            #max_indices = torch.argmax(max_of_all_likelihoods, dim=1)
            max_indices = torch.argmax(boxes[:, :, 4], dim=1)
            log.info(np.unique(max_indices.cpu().numpy(), return_counts=True))
            boxes = boxes[torch.arange(batch_size), max_indices]
            boxes_NT = boxes_NT[torch.arange(batch_size), max_indices]
            all_boxes = [boxes, boxes_NT]

        max_l5 = torch.max(l5[torch.arange(batch_size), pred], dim=1)[0]
        max_l4 = torch.max(l4[torch.arange(batch_size), pred], dim=1)[0]
        #max_l3 = torch.max(l3[torch.arange(batch_size), pred], dim=1)[0]
        #max_l5, max_l4 = torch.max(l5.view(batch_size, -1), dim=1)[0], torch.max(l4.view(batch_size, -1), dim=1)[0]
        reg_score = torch.sum(torch.max(torch.zeros(max_l5.shape).cuda(), (max_l4 - max_l5)))
        #reg_score += torch.sum(torch.max(torch.zeros(max_l5.shape).cuda(), (max_l3 - max_l4)))
        return likelihood, all_boxes, theta_diff, reg_score

