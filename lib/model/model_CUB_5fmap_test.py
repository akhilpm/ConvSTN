import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils import data_utils
from utils import vis_utils
#from torchvision.models import vgg16
#from torchvision.models import vgg16_bn
from torchvision.models import resnet101
import seaborn as sns
import math

#from utils.nms import nms
#from utils import box_utils
from utils import box_utils_new
from utils import box_utils_cub

#fmap = 4
fmap = 5
H = W = 320
vggnet = [["conv1_1", 3, 1, 1], ["conv1_2", 3, 1, 1], ["pool1", 2, 2, 0], ["conv2_1", 3, 1, 1], ["conv2_2", 3, 1, 1],
          ["pool2", 2, 2, 0], ["conv3_1", 3, 1, 1], ["conv3_2", 3, 1, 1], ["conv3_3", 3, 1, 1], ["pool3", 2, 2, 0],
         ["conv4_1", 3, 1, 1], ["conv4_2", 3, 1, 1], ["conv4_3", 3, 1, 1], ["pool4", 2, 2, 0], ["conv5_1", 3, 1, 1],
         ["conv5_2", 3, 1, 1], ["conv5_3", 3, 1, 1], ["pool5", 2, 2, 0]]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.model = vgg16(pretrained=True).cuda()
        #self.model = vgg16_bn(pretrained=True).cuda()
        self.model = resnet101(pretrained=True).cuda()
        self.n_classes = 200
        #self.conv_stn = nn.Conv2d(512, self.n_classes+1, kernel_size=3, stride=3)
        #self.conv_stn = nn.Conv2d(512, self.n_classes+1, kernel_size=3, padding=3/2)
        #self.conv_stn = nn.Conv2d(2048, 201, kernel_size=3)
        self.conv_stn = nn.Conv2d(2048, self.n_classes+1, kernel_size=3, stride=3)
        #self.conv_stn = nn.Conv2d(2048, 1024, kernel_size=3, stride=3)
        #self.conv = nn.Conv2d(1024, self.n_classes+1, kernel_size=1, stride=1)
        #self.conv = nn.Conv2d(2048, self.n_classes+1, kernel_size=3, stride=1, padding=3/2)


        #set up grid positions in the begining to parallelize apply
        self.pos_array = torch.zeros([fmap**2, 3, 3, 2]).cuda()
        self.no_transform = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).reshape(1,2,3).cuda()
        self.temp_var = torch.randn(1, 1, fmap, fmap)
        self.pad = torch.nn.ConstantPad2d(3/2, 0)
        self.temp_var = self.pad(self.temp_var)
        self.grid_positions_fs = F.affine_grid(self.no_transform, self.temp_var.size())
        index = 0
        for i in range(1, fmap+1):
            for j in range(1, fmap+1):
                self.pos_array[index] = self.grid_positions_fs[0, i-1:i+2, j-1:j+2, :]
                index += 1

        self.temp_var = torch.randn(1, 1, H, W)
        self.grid_positions_is = F.affine_grid(self.no_transform, self.temp_var.size())

        # Spatial transformer localization-network
        #for ResNet-101
        self.localization = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, padding=3/2, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(True),
            nn.Conv2d(2048, 2048, kernel_size=3, padding=3/2, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        #for ResNet-101
        self.fc_loc = nn.Sequential(
            nn.Linear(2048*9, 512, bias=False),
            nn.ReLU(True),
            nn.Linear(512, 2*2, bias=False)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        #self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        #print(model._modules) #list all layers in vgg16
        #self.features = nn.Sequential(*list(self.model.features.children()))
        #self.features = self.features[:38] #when using vg16_bn
        #self.features = self.features[:26] #when using vgg16
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        #self.conv_last = nn.Conv2d(2048, 2048, kernel_size=(1,1), stride=(2,2), bias=False)
        self.conv_last = nn.Conv2d(2048, 2048, kernel_size=3, stride=2, padding=3/2, bias=False)
        self.bn_last = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True)

        # Fix the layers before conv3: for VGG
        # layers upto 5(not including 5) for ResNet-101
        for layer in range(5):
            for p in self.features[layer].parameters(): p.requires_grad = False


        #compute the receptive field of a 3*3 block around each pixel in the feature map.
        self.unit = H/float(fmap+1)
        self.box_pos = np.zeros([fmap, fmap, 4])
        #f =  open('../boxes_pos.txt', 'w')
        #for idx_y in range(fmap):
        #    for idx_x in range(fmap):
        #        min_x, min_y = max(idx_x-1, 0), max(idx_y-1, 0)
        #        max_x, max_y = min(idx_x+2, fmap), min(idx_y+2, fmap)
        #        self.box_pos[idx_y, idx_x, 0:2] = np.clip(np.array([self.unit*min_x, self.unit*min_y]), 0, H-1)
        #        self.box_pos[idx_y, idx_x, 2:4] = np.clip(np.array([self.unit*max_x, self.unit*max_y]), 0, H-1)
        for idx_y in range(1, fmap+1):
            for idx_x in range(1, fmap+1):
                min_x, min_y = max(idx_x-1, 0), max(idx_y-1, 0)
                max_x, max_y = min(idx_x+2, fmap+1), min(idx_y+2, fmap+1)
                self.box_pos[idx_y-1, idx_x-1, 0:2] = np.clip(np.array([self.unit*min_x, self.unit*min_y]), 0, H-1)
                self.box_pos[idx_y-1, idx_x-1, 2:4] = np.clip(np.array([self.unit*max_x, self.unit*max_y]), 0, H-1)


    # Spatial transformer network forward function
    def stn(self, x, flag, batch_id):
        batch_size, channels, h, w = x.size()
        xs = self.localization(x)
        xs = self.pad(xs)
        #no_transform = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).reshape(2, 3).cuda()
        #theta = torch.zeros(batch_size * fmap * fmap, 2, 3).cuda()
        #theta.copy_(no_transform)
        theta = torch.zeros([batch_size, fmap, fmap, 2, 3]).cuda()
        for i in range(1, fmap+1):
            for j in range(1, fmap+1):
                stn_input = xs[:, :, i-1:i+2, j-1:j+2]
                stn_input = stn_input.contiguous().view(-1, 2048*3*3)
                params = self.fc_loc(stn_input)
                no_rotation = torch.zeros(params[:, 1].shape).cuda()
                list_of_tensors = [params[:, 0], no_rotation, params[:, 1], no_rotation,
                                   params[:, 2], params[:, 3]]
                theta_ij = torch.t(torch.stack(list_of_tensors))
                theta_ij = theta_ij.view(-1, 2, 3) + self.no_transform
                theta[:, i-1, j-1] = theta_ij
        theta = theta.view(batch_size*(fmap**2), 2, 3)

        #create grid coordinates for all images in the batch
        pos_array = self.pos_array.repeat(batch_size, 1, 1, 1)
        pos = pos_array.contiguous().view(-1, 2)
        pos = torch.cat([pos, torch.ones(batch_size*(fmap**2)*9, 1).cuda()], dim=1) # "9" because at each pos 3*3 block is considered
        pos = pos.view(batch_size*(fmap**2), 9, 3)
        prod = torch.bmm(pos, theta.permute(0,2,1))
        prod = prod.view(batch_size*(fmap**2), 3, 3, 2)

        count = 0
        max_idx = 3*h-1 #since 3*3 block is considered everywhere
        temp = torch.zeros([batch_size, channels, 3*h, 3*w]).cuda()
        x = self.pad(x)

        for i in range(1, max_idx, 3):
            for j in range(1, max_idx, 3):
                temp[:, :, i-1:i+2, j-1:j+2] = F.grid_sample(x, prod[count::fmap*fmap])
                count += 1

        temp = self.conv_stn(temp)
        #temp = F.relu(self.conv_stn(temp))
        return temp, theta



    def forward(self, x, epoch, flag, log, batch_id=0, batch_size=4):
        #data = x.clone()
        batch_size = x.shape[0]

        # Perform the usual forward pass
        #x = self.features(x)
        x = self.bn_last(self.conv_last(self.features(x)))
        #x_noSTN = self.conv(x)
        #x_noSTN = x_noSTN.contiguous().view(-1, fmap*fmap*(self.n_classes+1))
        #x_noSTN = F.softmax(x_noSTN, dim=1)
        #x_noSTN = x_noSTN.view(-1, self.n_classes+1, fmap, fmap)

        # transform the input
        #with torch.no_grad():
        x_STN, theta = self.stn(x, flag, batch_id)
        #x_STN = self.conv_stn(x_STN)

        #reshape it to a 1D vector, apply softmax and get it back to the same shape
        x = x_STN.contiguous().view(-1, fmap*fmap*(self.n_classes+1))
        x = F.softmax(x, dim=1)
        x = x.view(-1, self.n_classes+1, fmap, fmap)

        classif_scores = x.view(batch_size, self.n_classes+1, -1).sum(2)

        no_transform = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).reshape(2,3).cuda()
        theta_NT = torch.zeros(batch_size*fmap*fmap, 2, 3).cuda()
        theta_NT.copy_(no_transform)
        with torch.no_grad():
            boxes_max = box_utils_cub.get_transformed_bounding_box_cub(self.box_pos, theta, classif_scores, batch_size,
                self.grid_positions_is[0].detach(), H, x)
            #boxes_all = box_utils_new.get_transformed_bounding_box_cub(self.box_pos, theta, classif_scores, batch_size,
#self.grid_positions_is[0].detach(), H, x)
            boxes_max_NT = box_utils_cub.get_transformed_bounding_box_cub(self.box_pos, theta_NT, classif_scores, batch_size,
                self.grid_positions_is[0].detach(), H, x_STN)
            #boxes_all_NT = box_utils_new.get_transformed_bounding_box_cub(self.box_pos, theta_NT, classif_scores, batch_size,
#self.grid_positions_is[0].detach(), H, x)
            #all_boxes = [boxes_max, boxes_all, boxes_max_NT, boxes_all_NT]
            all_boxes = [boxes_max, 0, boxes_max_NT, 0]
        #all_boxes = torch.zeros(batch_size, 5)
        return classif_scores, x, all_boxes, theta_NT
        #return classif_scores, x, x_STN, all_boxes, theta
