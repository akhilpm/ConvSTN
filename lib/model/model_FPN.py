import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
from torchvision.models import resnet101
from model.model_convSTN_FPN import Conv2d_STN

#fmap = 4
fmap = 5
H = W = 320

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = resnet101(pretrained=True).cuda()
        self.n_classes = 201

        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.conv_last_10map = nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=3/2)
        self.bn_last_10map = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True)

        # Bottom-up layers
        self.layer1 = self.features[:5]  # 80*80 fmap for 320*320 input - 28*28 RF for 7*7 kernel
        self.layer2 = self.features[5]  # 40*40 fmap for 320*320 input - 56*56 RF for 7*7 kernel
        self.layer3 = self.features[6]  # 20*20 fmap for 320*320 input - 112*112 RF for 7*7 kernel
        self.layer4 = self.features[7]  # 10*10 fmap for 320*320 input - 224*224 RF for 7*7 kernel


        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        # Conv_STN
        self.stn = Conv2d_STN(256, 256, 7)
        #self.stn = Conv2d_STN(1024, 1024, 7)

        # Fix the layers before conv3: for VGG
        # layers upto 5(not including 5) for ResNet-101
        for layer in range(5):
            for p in self.features[layer].parameters(): p.requires_grad = False


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y


    def _upsample(self, x, H, W):
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)


    def forward(self, x, image_dim, epoch, log, batch_id=0, batch_size=4):

        # Bottom-up
        c2 = self.layer1(x)     # 80*80/256
        c3 = self.layer2(c2)    # 40*40/512
        c4 = self.layer3(c3)    # 20*20/1024
        c3 = self.latlayer2(c3) # 20*20/256
        c5 = self.layer4(c4)    # 10*10/2048
        c4 = self.latlayer1(c4) # 10*10/256
        c5 = self.bn_last_10map(self.conv_last_10map(F.relu(c5)))

        # Top-down
        p5 = self.toplayer(F.relu(c5))
        midH = np.int((c5.shape[2]+c4.shape[2])/2)
        mid_p5 = self._upsample(p5, midH, midH)
        #p4 = self._upsample_add(p5, c4)
        #p4 = self._upsample_add(mid_p5, c4)
        #p3 = self._upsample_add(p4, c3)
        #p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        #p4 = self.smooth1(p4)
        #p3 = self.smooth2(p3)
        #p2 = self.smooth3(p2)
        del c2, c3, c4, c5
        gc.collect()



        # transform the input and get boxes, likelihood etc
        likelihood, all_boxes, theta_diff, reg_score = self.stn([p5, mid_p5], log, image_dim=image_dim)
        #likelihood, all_boxes, theta_diff, reg_score = self.stn([p5, p4], log, image_dim=image_dim)
        return likelihood, all_boxes, theta_diff, reg_score
