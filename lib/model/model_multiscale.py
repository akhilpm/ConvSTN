import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101
from model.temp import Conv2d_STN


class Net(nn.Module):
    def __init__(self, n_classes):
        super(Net, self).__init__()
        self.model = resnet101(pretrained=True).cuda()
        self.num_classes = n_classes
        self.kernel1_size = 7
        self.kernel2_size = 13

        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.conv_last_10map = nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=3/2)
        self.bn_last_10map = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True)
        self.layer1 = self.features[:7]
        self.layer2 = self.features[7]
        #self.conv_last_10map = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=3/2)
        #self.bn_last_10map = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True)
        self.conv_last = nn.Conv2d(512, self.num_classes, kernel_size=1)
        #self.conv_last2 = nn.Conv2d(512, self.num_classes, kernel_size=1)

        self.toplayer = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0)  # Reduce channels
        self.latlayer1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.smooth1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        #self.stn =  Conv2d_STN(2048, self.n_classes, 5)
        self.stn1 = Conv2d_STN(512, 512, self.kernel1_size)
        self.stn2 = Conv2d_STN(512, 512, self.kernel2_size)

        # Fix the layers before conv3: for VGG
        # layers upto 5(not including 5) for ResNet-101
        for layer in range(5):
            for p in self.features[layer].parameters(): p.requires_grad = False

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def get_no_transform_box(self, kernel_size, theta, x, out, detection_locs, image_dim, conf):
        no_transform = torch.tensor([1., 0., 0., 0., 1., 0.]).cuda().view(1,2,3)
        theta_NT = torch.zeros(theta.shape).cuda().copy_(no_transform)
        grid = F.affine_grid(theta_NT, (theta.shape[0], x.shape[1], kernel_size, kernel_size))
        xv, yv = torch.meshgrid(
        [torch.arange(kernel_size / 2, kernel_size / 2 + out.shape[2]),
         torch.arange(kernel_size / 2, kernel_size / 2 + out.shape[3])])
        auxx = ((grid[:, :, :, 0] * (kernel_size / 2)).view(out.shape[0], out.shape[2], out.shape[3],
                 kernel_size, kernel_size) + xv.view(1, out.shape[2], out.shape[3], 1, 1).cuda().float()).view(
            out.shape[0] * out.shape[2] * out.shape[3], kernel_size, kernel_size) / (x.shape[2] - 1) * 2 - 1
        auxy = ((grid[:, :, :, 1] * (kernel_size / 2)).view(out.shape[0], out.shape[2], out.shape[3],
             kernel_size, kernel_size) + yv.view(1, out.shape[2], out.shape[3], 1, 1).cuda().float()).view(
            out.shape[0] * out.shape[2] * out.shape[3], kernel_size, kernel_size) / (x.shape[3] - 1) * 2 - 1

        tempx, tempy = (auxx + 1) * 0.5, (auxy + 1) * 0.5
        tempx, tempy = tempx.view(theta.shape[0], -1), tempy.view(theta.shape[0], -1)
        min_x, min_y = (torch.min(tempx[detection_locs], dim=1)[0] * image_dim).clamp(0, image_dim - 1), (
                    torch.min(tempy[detection_locs], dim=1)[0] * image_dim).clamp(0, image_dim - 1)
        max_x, max_y = (torch.max(tempx[detection_locs], dim=1)[0] * image_dim).clamp(0, image_dim - 1), (
                    torch.max(tempy[detection_locs], dim=1)[0] * image_dim).clamp(0, image_dim - 1)
        box = torch.stack([min_x, min_y, max_x, max_y, conf], dim=1)
        return box

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

    def forward(self, x, image_dim, epoch, log, batch_id=0, batch_size=4):

        #x = F.relu(self.features(x))
        #x = self.bn_last_10map(self.conv_last_10map(x))
        #x =  self.toplayer(F.relu(x))

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c2 = self.bn_last_10map(self.conv_last_10map(F.relu(c2)))
        c2 = self.toplayer(c2)
        c1 = self.latlayer1(c1)
        c1 = self._upsample_add(c2, c1)
        c1 = self.smooth1(c1)

        # transform the input
        score_k1, theta_k1, out_k1, auxx_k1, auxy_k1 = self.stn1(c2)
        score_k1 = self.conv_last(F.relu(score_k1))
        score_k2, theta_k2, out_k2, auxx_k2, auxy_k2 = self.stn2(c1)
        score_k2 = self.conv_last(F.relu(score_k2))
        theta_diff = torch.cat([theta_k1, theta_k2], dim=0)
        score_k1 = score_k1.view(batch_size, self.num_classes, -1)
        score_k2 = score_k2.view(batch_size, self.num_classes, -1)
        score = torch.cat([score_k1, score_k2], dim=2)

        joint = F.softmax(score.view(batch_size, -1), dim=1).view(batch_size, self.num_classes, -1)
        likelihood =  joint.sum(2)
        k1 = (joint[:, :, :score_k1.shape[2]]).contiguous().view(batch_size, self.num_classes, -1)
        k2 = (joint[:, :, score_k1.shape[2]:score_k1.shape[2]+score_k2.shape[2]]).contiguous().view(batch_size, self.num_classes, -1)
        pred = torch.argmax(likelihood, dim=1)
        #max_box_pos = joint[torch.arange(batch_size), pred]
        #max_box_pos = torch.max(max_box_pos, dim=1)[0]
        box_k1, detection_locs, conf = self.get_max_boxes(k1, pred, out_k1, auxx_k1, auxy_k1, image_dim)
        box_NT_k1 = self.get_no_transform_box(self.kernel1_size, theta_k1, c2, out_k1, detection_locs, image_dim, conf)
        box_k2, detection_locs, conf = self.get_max_boxes(k2, pred, out_k2, auxx_k2, auxy_k2, image_dim)
        box_NT_k2 = self.get_no_transform_box(self.kernel2_size, theta_k2, c1, out_k2, detection_locs, image_dim, conf)
        boxes = torch.cat([torch.unsqueeze(box_k1, dim=1), torch.unsqueeze(box_k2, dim=1)], dim=1)
        boxes_NT = torch.cat([torch.unsqueeze(box_NT_k1, dim=1), torch.unsqueeze(box_NT_k2, dim=1)], dim=1)

        max_indices = torch.argmax(boxes[:, :, 4], dim=1)
        log.info(np.unique(max_indices.cpu().numpy(), return_counts=True))
        boxes = boxes[torch.arange(batch_size), max_indices]
        boxes_NT = boxes_NT[torch.arange(batch_size), max_indices]
        all_boxes = [boxes, boxes_NT]

        max_k1 = torch.max(k1[torch.arange(batch_size), pred], dim=1)[0]
        max_k2 = torch.max(k2[torch.arange(batch_size), pred], dim=1)[0]
        #max_k1, max_k2 = torch.max(k1.view(batch_size, -1), dim=1)[0], torch.max(k2.view(batch_size, -1), dim=1)[0]
        reg_score = torch.sum(torch.max(torch.zeros(max_k1.shape).cuda(), (max_k2 - max_k1)))
        return likelihood, all_boxes, theta_diff, reg_score