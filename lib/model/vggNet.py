import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils import data_utils
from utils import vis_utils
from torchvision.models import vgg16
import seaborn as sns

from utils.nms import nms
from utils import box_utils

fmap = 16
H = W = 512

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.fc1 = nn.Linear(4*4*512, 512)
        #self.fc2 = nn.Linear(512, 10)
        self.fc_drop = nn.Dropout()
        self.model = vgg16(pretrained=True).cuda()
        self.n_classes = 20
        self.conv_stn = nn.Conv2d(512, self.n_classes+1, kernel_size=3, stride=3)
        #self.conv_stn = nn.Conv2d(512, 256, kernel_size=3, stride=3)
        #self.class_imp = nn.Conv2d(512, self.n_classes, kernel_size=3, padding=3/2)
        #self.fc1 = nn.Linear(4*4*256, 512)
        #self.fc2 = nn.Linear(512, 10)


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

        self.temp_var = torch.randn(1, 1, 500, 500)
        self.grid_positions_is = F.affine_grid(self.no_transform, self.temp_var.size())

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=3/2),
            #nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(512, 512, kernel_size=3, padding=3/2),
            #nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout2d(p=0.5)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.Linear(64, 3*2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        #print(model._modules) #list all layers in vgg16
        self.features = nn.Sequential(*list(self.model.features.children()))

        # Fix the layers before conv3:
        for layer in range(10):
            for p in self.model.features[layer].parameters(): p.requires_grad = False
        #freeze the feature extraction layers
        #for param in self.features.parameters():
        #    param.requires_grad = False




    # Spatial transformer network forward function
    def stn(self, data, x, epoch, flag, batch_id, log):
        batch_size, channels, h, w = x.size()
        xs = self.localization(x)
        xs = xs.permute(0,2,3,1)
        xs = xs.contiguous().view(-1, 512)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        #theta[:, 0, 1] = theta[:, 1, 0] = 0.0

        #constrain transformation to have no rotation
        #rot_pos = [torch.tensor([0, 1]).cuda(), torch.tensor([1, 0]).cuda()]
        #_ = [item.index_put_(rot_pos, torch.zeros(2).cuda()) for item in theta]
        theta_xclip = torch.clamp(theta[:, 0, 1], min=-0.0001, max=0.0001)
        theta_yclip = torch.clamp(theta[:, 1, 0], min=-0.0001, max=0.0001)
        list_of_tensors = [theta[:, 0, 0], theta_xclip, theta[:, 0, 2], theta_yclip,
                            theta[:, 1, 1], theta[:, 1, 2]]
        theta = torch.t(torch.stack(list_of_tensors))
        theta = theta.view(-1, 2, 3)
        #print(theta.shape)

        x = self.pad(x)
        """ set up grid positions according to the feature map dimension. The transformation is applied at each
        position in the grid by taking a 3*3 window around that position in a convolutional fashion """
        #with torch.no_grad():
        #    #print(type(no_transform[0][1][1]))
        #    no_transform = data_utils.tile(self.no_transform, 0, batch_size)
        #    grid_positions_fs = F.affine_grid(no_transform, x.size())
        #    grid_positions_is = F.affine_grid(no_transform, data.size())

        #temp = x.clone().data.fill_(0)
        #create grid coordinates for all images in the batch
        pos_array = self.pos_array.repeat(batch_size, 1, 1, 1)
        pos = pos_array.contiguous().view(-1, 2)
        pos = torch.cat([pos, torch.ones(batch_size*(fmap**2)*9, 1).cuda()], dim=1)
        pos = pos.view(batch_size*(fmap**2), 9 ,3)
        prod = torch.bmm(pos, theta.permute(0,2,1))
        #print("theta, cord prod shape: {}".format(prod.shape))

        prod = prod.view(batch_size*(fmap**2), 3, 3, 2)
        #print(prod.shape)
        #if flag=="test" and batch_id==1:
        #    print("+"*50)
        #    print(prod[:fmap**2])
        #    print("+"*50)

        count=0
        max_idx = 3*h-1
        temp = torch.zeros([batch_size, channels, 3*h, 3*w]).cuda()
        #print(temp.shape)

        for i in range(1, max_idx, 3):
            for j in range(1, max_idx, 3):
                #pixel_intensities = xs[:, :, i, j]

                #extract anchor box at each position and transform them
                #cord = self.apply_transform(theta[count::fmap*fmap], grid_positions_fs[:, i-1:i+2, j-1:j+2, :])
                #print(cord.shape)
                #grid = F.affine_grid(theta, x[:, :, i:i+3, j:j+3].size())
                #v1 = F.grid_sample(x, cord)
                #v2 = F.grid_sample(x, prod[count::fmap*fmap])
                #print(torch.sum(v1==v2))
                #temp[:, :, i-1:i+2, j-1:j+2] = F.grid_sample(x, cord)
                temp[:, :, i-1:i+2, j-1:j+2] = F.grid_sample(x, prod[count::fmap*fmap])
                count += 1

        temp = self.conv_stn(temp)
        return temp, theta



    def forward(self, x, epoch, flag, log, batch_id=0, batch_size=64):
        data = x.clone()

        # Perform the usual forward pass
        x = self.features(x)
        #print(torch.norm(x))
        #print(x.shape)
        #x = x.contiguous().view(-1, 4*4*128)
        #x = F.softmax(x, dim=1)
        #x = x.view(-1, 128, 4, 4)

        # transform the input
        x, theta = self.stn(data, x, epoch, flag, batch_id, log)
        #print(torch.norm(x))
        #print(self.fc_loc[2].weight.data.norm())

        #with FC reduction
        """
        x = x.contiguous().view(-1, 4*4*256)
        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        logits = F.relu(self.fc2(x))

        """ 
        #with conv reduction

        #reduce the number of channels to n_classes
        #x = self.conv1_red(x)
        #x = self.class_imp(x)
        #print(x.shape, data.shape, theta.shape)
        #reshape it to a 1D vector, apply softmax and get it back to the same shape
        x = x.contiguous().view(-1, fmap*fmap*(self.n_classes+1))
        x = F.softmax(x, dim=1)
        x = x.view(-1, self.n_classes+1, fmap, fmap)

        classif_scores = x.view(batch_size, self.n_classes+1, -1).sum(2)
        #"""
        if flag=="test" and batch_id==1:
            with torch.no_grad():
                detection_scores = x.sum(1)
                vis_utils.plot_receptive_field_transform_single_image(theta, data[1], self.grid_positions_is[0], 
                    512, 512, epoch, fmap, log, img_indx=1)
                #detection_scores = torch.ones(64, 4, 4)*0.5
                #vis_utils.plot_sampling_points_local_transform(theta, data, self.grid_positions_is[0], 
                #    512, 512, epoch, fmap, log, detection_scores)
                #print("="*30)
                #print(theta[:4])
                #print("="*30)
                ax = sns.heatmap(detection_scores[1].cpu().numpy(), linewidth=0.5)
            plt.savefig("results/class_heatmap/det_scores_"+str(epoch)+".png")
            plt.clf()
            plt.close()

        if flag=="test":
            final_detection = []
            with torch.no_grad():
                all_boxes = box_utils.get_bb_after_transform(theta, data, H, W, epoch, fmap)
                for i in range(1, self.n_classes+1):
                    all_boxes[:, :, 4] = x[:, i].view(data.shape[0], fmap**2)
                    selected_boxes = [nms(all_boxes[j], 0.4) for j in all_boxes.shape[0]]
                    final_detection.append(selected_boxes)


        if flag=="train":
            all_boxes = final_detection = 0
            #print("classification scores[1]:\n {}".format(classif_scores[1]))
            #print("detection scores[1]:\n {}".format(detection_scores[1]))
        #return F.log_softmax(logits, dim=1), F.softmax(logits, dim=1)
        return classif_scores, x, all_boxes, final_detection
