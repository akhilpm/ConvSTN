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

fmap = 10
#fmap = 5
H = W = 320
vggnet = [["conv1_1", 3, 1, 1], ["conv1_2", 3, 1, 1], ["pool1", 2, 2, 0], ["conv2_1", 3, 1, 1], ["conv2_2", 3, 1, 1], 
          ["pool2", 2, 2, 0], ["conv3_1", 3, 1, 1], ["conv3_2", 3, 1, 1], ["conv3_3", 3, 1, 1], ["pool3", 2, 2, 0],
         ["conv4_1", 3, 1, 1], ["conv4_2", 3, 1, 1], ["conv4_3", 3, 1, 1], ["pool4", 2, 2, 0], ["conv5_1", 3, 1, 1],
         ["conv5_2", 3, 1, 1], ["conv5_3", 3, 1, 1], ["pool5", 2, 2, 0]]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.fc1 = nn.Linear(4*4*512, 512)
        #self.fc2 = nn.Linear(512, 10)
        self.fc_drop = nn.Dropout(p=0.5)
        #self.model = vgg16(pretrained=True).cuda()
        #self.model = vgg16_bn(pretrained=True).cuda()
        self.model = resnet101(pretrained=True).cuda()
        self.n_classes = 200
        #self.conv_stn = nn.Conv2d(512, self.n_classes+1, kernel_size=3, stride=3)
        #self.conv_stn = nn.Conv2d(512, self.n_classes+1, kernel_size=3, padding=3/2)
        #self.conv_stn = nn.Conv2d(2048, 201, kernel_size=3)
        #self.conv_stn = nn.Conv2d(2048, self.n_classes+1, kernel_size=3, stride=3)
        #self.conv_stn = nn.Conv2d(2048, 1024, kernel_size=3, stride=3)
        #self.conv_stn = nn.Conv2d(2048, self.n_classes+1, kernel_size=1, stride=1)
        self.conv_stn = nn.Conv2d(2048, 1024, kernel_size=3, padding=3/2)
        self.conv_last = nn.Conv2d(1024, self.n_classes+1, kernel_size=1, stride=1)
        #self.conv_stn = nn.Conv2d(2048, self.n_classes+1, kernel_size=1, stride=1)
        #self.class_imp = nn.Conv2d(512, self.n_classes, kernel_size=3, padding=3/2)
        #self.fc1 = nn.Linear(4*4*256, 512)
        #self.fc2 = nn.Linear(512, 10)
        #self.GAP = nn.AvgPool2d(kernel_size=5)
        #self.fc_GAP = nn.Linear(512*4*4, self.n_classes+1)


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
        #for VGG
        """
        self.localization = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=3/2),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(512, 512, kernel_size=3, padding=3/2),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout2d(p=0.5)
        )
        """
        #for ResNet-101
        self.localization = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, padding=3/2),
            nn.BatchNorm2d(2048),
            nn.ReLU(True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(2048, 2048, kernel_size=3, padding=3/2),
            nn.BatchNorm2d(2048),
            nn.ReLU(True),
            nn.Dropout2d(p=0.5)
        )

        # Regressor for the 3 * 2 affine matrix
        #for vgg16
        """
        self.fc_loc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.Linear(64, 3*2)
        )
        """
        #for ResNet-101
        self.fc_loc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Linear(512, 3*2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        #print(model._modules) #list all layers in vgg16
        #self.features = nn.Sequential(*list(self.model.features.children()))
        #self.features = self.features[:38] #when using vg16_bn
        #self.features = self.features[:26] #when using vgg16
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        #self.conv_last = nn.Conv2d(2048, 2048, kernel_size=(1,1), stride=(2,2), bias=False)
        #self.bn_last = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True)

        # Fix the layers before conv3: for VGG
        # layers upto 5(not including 5) for ResNet-101
        for layer in range(5):
            for p in self.features[layer].parameters(): p.requires_grad = False
        #freeze the feature extraction layers
        #for param in self.features.parameters():
        #    param.requires_grad = False

        #currentLayer = [H, 1, 1, 0]
        #self.layerInfos = []
        #for i in range(len(vggnet)):
        #    currentLayer = self.outFromIn(vggnet[i], currentLayer)
        #    self.layerInfos.append(currentLayer)
        #layer_names = [layer[0] for layer in vggnet]
        #layer_idx = layer_names.index("conv5_1")
        #self.rf = self.layerInfos[layer_idx][2] #size of the receptive field size
        #self.centers = np.zeros([fmap, fmap, 2])
        #self.rf_box_pos = np.zeros([fmap, fmap, 4])
        #jump = self.layerInfos[layer_idx][1]
        #start = self.layerInfos[layer_idx][3]
        #compute the receptive field coordinates of each point in the feature map pixel
        #for idx_y in range(fmap):
        #    for idx_x in range(fmap):
        #        self.centers[idx_y, idx_x] = [start+idx_x*jump, start+idx_y*jump]
        #        self.rf_box_pos[idx_y, idx_x, 0:2] = np.clip(self.centers[idx_y, idx_x] - self.rf/2.0, 0, H-1)
        #        self.rf_box_pos[idx_y, idx_x, 2:4] = np.clip(self.centers[idx_y, idx_x] + self.rf/2.0, 0, H-1)
                #print(self.rf_box_pos[idx_y, idx_x])

        #compute the receptive field of a 3*3 block around each pixel in the feature map.
        self.unit = H/float(fmap)
        self.box_pos = np.zeros([fmap, fmap, 4])
        #f =  open('../boxes_pos.txt', 'w')
        for idx_y in range(fmap):
            for idx_x in range(fmap):
                min_x, min_y = max(idx_x-1, 0), max(idx_y-1, 0)
                max_x, max_y = min(idx_x+2, fmap), min(idx_y+2, fmap)
                #self.box_pos[idx_y, idx_x, 0:2] = self.rf_box_pos[min_y, min_x, 0:2]
                #self.box_pos[idx_y, idx_x, 2:4] = self.rf_box_pos[max_y, max_x, 2:4]
                self.box_pos[idx_y, idx_x, 0:2] = np.clip(np.array([self.unit*min_x, self.unit*min_y]), 0, H-1)
                self.box_pos[idx_y, idx_x, 2:4] = np.clip(np.array([self.unit*max_x, self.unit*max_y]), 0, H-1)
                #f.write(str(self.box_pos[idx_y, idx_x])+"\n")
                #print(self.box_pos[idx_y, idx_x])
        #f.close()
        #print(self.centers)


    def outFromIn(self, conv, layerIn):
        #Each kernel requires the following parameters:
        # - k_i: kernel size
        # - s_i: stride
        # - p_i: padding (if padding is uneven, right padding will higher than left padding; "SAME" option in tensorflow)
        # 
        #Each layer i requires the following parameters to be fully represented: 
        # - n_i: number of feature (data layer has n_1 = imagesize )
        # - j_i: distance (projected to image pixel distance) between center of two adjacent features
        # - r_i: receptive field of a feature in layer i
        # - start_i: position of the first feature's receptive field in layer i (idx start from 0, negative means the center fall into padding)
        n_in = layerIn[0]
        j_in = layerIn[1]
        r_in = layerIn[2]
        start_in = layerIn[3]
        k = conv[1]
        s = conv[2]
        p = conv[3]

        n_out = math.floor((n_in - k + 2*p)/s) + 1
        actualP = (n_out-1)*s - n_in + k
        pR = math.ceil(actualP/2)
        pL = math.floor(actualP/2)

        j_out = j_in * s
        r_out = r_in + (k - 1)*j_in
        start_out = start_in + ((k-1)/2 - pL)*j_in
        return n_out, j_out, r_out, start_out


    # Spatial transformer network forward function
    def stn(self, x, epoch, flag, batch_id, log):
        batch_size, channels, h, w = x.size()
        xs = self.localization(x)
        xs = xs.permute(0,2,3,1)
        #xs = xs.contiguous().view(-1, 512)
        xs = xs.contiguous().view(-1, 2048)
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
        if epoch==1 and batch_id==1 and flag=="train":
            pass
        else:
            theta += self.no_transform
        #if epoch != 1:
        #    theta += self.no_transform
        #theta.data.copy_(torch.tensor([1,0,0,0,1,0], dtype=torch.float).reshape(2,3))
        #print(theta.shape)

        x = self.pad(x)

        #temp = x.clone().data.fill_(0)
        #create grid coordinates for all images in the batch
        pos_array = self.pos_array.repeat(batch_size, 1, 1, 1)
        pos = pos_array.contiguous().view(-1, 2)
        pos = torch.cat([pos, torch.ones(batch_size*(fmap**2)*9, 1).cuda()], dim=1) # "9" because at each pos 3*3 block is considered
        pos = pos.view(batch_size*(fmap**2), 9, 3)
        prod = torch.bmm(pos, theta.permute(0,2,1))
        #print("theta, cord prod shape: {}".format(prod.shape))

        prod = prod.view(batch_size*(fmap**2), 3, 3, 2)

        count=0
        max_idx = 3*h-1 #since 3*3 block is considered everywhere
        #max_idx = h+1
        temp = torch.zeros([batch_size, channels, 3*h, 3*w]).cuda()
        #temp = torch.zeros(x.shape).cuda()
        #temp = self.pad(temp)
        #print(temp.shape)

        for i in range(1, max_idx, 3):
            for j in range(1, max_idx, 3):
                temp[:, :, i-1:i+2, j-1:j+2] = F.grid_sample(x, prod[count::fmap*fmap])
                count += 1

        #for i in range(1, max_idx):
        #    for j in range(1, max_idx):
        #        temp[:, :, i-1:i+2, j-1:j+2] += F.grid_sample(x, prod[count::fmap*fmap])
        #        count += 1
        temp = self.conv_stn(temp)
        #temp = self.conv_stn2(temp)
        #temp = F.relu(self.conv_stn(temp))
        #temp = self.conv(temp)
        return temp, theta



    def forward(self, x, epoch, flag, log, batch_id=0, batch_size=4):
        #data = x.clone()
        batch_size = x.shape[0]
        #print("="*50)
        #print(x.shape)
        #print("="*50)

        # Perform the usual forward pass
        x = self.features(x)
        #x = self.bn_last(self.conv_last(self.features(x)))
        #x = self.conv_stn(x)
        #x = self.GAP(x)
        #x = x.view(batch_size, -1)
        #x = self.fc_drop(x)
        #x = self.fc_GAP(x)
        #classif_scores = F.softmax(x, dim=1)
        #print(torch.norm(x))
        #print(x.shape)
        #x = x.contiguous().view(-1, 4*4*128)
        #x = F.softmax(x, dim=1)
        #x = x.view(-1, 128, 4, 4)

        # transform the input
        #x, theta = self.stn(x, epoch, flag, batch_id, log)
        x = self.conv_stn(x)
        x = self.conv_last(F.relu(x))
        #print(x.shape)
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

        """
        if flag=="test" and batch_id%100==0:
            with torch.no_grad():
                detection_scores = x.sum(1)
                #vis_utils.plot_receptive_field_transform_single_image(self.box_pos, theta, data[1], self.grid_positions_is[0], 
                #    H, W, epoch, batch_id, fmap, log, detection_scores, flag, img_indx=1)
                #detection_scores = torch.ones(64, 4, 4)*0.5
                #vis_utils.plot_sampling_points_local_transform(theta, data, self.grid_positions_is[0], 
                #    512, 512, epoch, fmap, log, detection_scores)
                #print("="*30)
                #print(theta[:4])
                #print("="*30)
                ax = sns.heatmap(detection_scores[1].cpu().numpy(), linewidth=0.5)
            plt.savefig("../results/class_heatmap/test/det_scores_test_"+str(epoch)+".png")
            plt.clf()
            plt.close()

        if flag=="train" and batch_id%100==0:
            with torch.no_grad():
                detection_scores = x.sum(1)
                #vis_utils.plot_receptive_field_transform_single_image(self.box_pos, theta, data[1], self.grid_positions_is[0], 
                #    H, W, epoch, batch_id, fmap, log, detection_scores, flag, img_indx=1)
                ax = sns.heatmap(detection_scores[1].cpu().numpy(), linewidth=0.5)
            plt.savefig("../results/class_heatmap/train/det_scores_train_"+str(epoch)+'_'+str(batch_id)+".png")
            plt.clf()
            plt.close()
        """
        no_transform = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).reshape(2,3).cuda()
        theta = torch.zeros(batch_size*fmap*fmap, 2, 3).cuda()
        theta.copy_(no_transform)
        with torch.no_grad():
            boxes_max = box_utils_cub.get_transformed_bounding_box_cub(self.box_pos, theta, classif_scores, batch_size, 
self.grid_positions_is[0].detach(), H, x)
            boxes_all = box_utils_new.get_transformed_bounding_box_cub(self.box_pos, theta, classif_scores, batch_size, 
self.grid_positions_is[0].detach(), H, x)
        all_boxes = [boxes_max, boxes_all]
        return classif_scores, x, all_boxes, theta
