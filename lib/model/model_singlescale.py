import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101
from torchvision.models.resnet import model_urls
from model.model_convSTN import Conv2d_STN

model_urls['resnet101'] = model_urls['resnet101'].replace('https://', 'http://')

class Net(nn.Module):
    def __init__(self, n_lasses):
        super(Net, self).__init__()
        self.model = resnet101(pretrained=True).cuda()
        self.n_classes = n_lasses

        self.features = nn.Sequential(*list(self.model.children())[:-3])
        #self.bottleneck1 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0)
        #self.bottleneck2 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0)
        #self.conv_last_10map = nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=3/2)
        #self.bn_last_10map = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True)
        self.conv_last_10map = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=3/2)
        self.bn_last_10map = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True)
        #self.conv_last = nn.Conv2d(2048, 2048, kernel_size=3, stride=2, padding=3/2)
        #self.maxpool = nn.MaxPool2d(2, 2)

        # Top layer
        #self.toplayer = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0)  # Reduce channels
        #self.latlayer = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        #self.smooth = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        #self.stn = Conv2d_STN(2048, 2048, 7, self.n_classes)
        self.stn = Conv2d_STN(1024, 1024, 13, self.n_classes)

        # Fix the layers before conv3: for VGG
        # layers upto 5(not including 5) for ResNet-101
        for layer in range(5):
            for p in self.features[layer].parameters(): p.requires_grad = False

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y


    def forward(self, x, image_res, epoch, log, batch_id=0, batch_size=4):

        # Perform the usual forward pass
        #c4 = self.features[:-1](x)
        #c4 = self.latlayer(c4)
        x = F.relu(self.features(x))
        x = self.bn_last_10map(self.conv_last_10map(x))
        #x = self.toplayer(x)
        #x = self._upsample_add(x, c4)
        #x = self.smooth(x)
        #x = self.maxpool(x)
        #x = self.conv_last(x)

        # transform the input
        likelihood, output, boxes, boxes_NT, theta_diff = self.stn(x, image_res)
        #conf = boxes[:, 4]
        all_boxes = [boxes, boxes_NT]
        return likelihood, output, all_boxes, theta_diff
