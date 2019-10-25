import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import data_utils
import logging
import sys
from PIL import Image
import os
import pickle
import glob
import os.path as osp
import gzip


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def sigmoid(x):
    return 1.0/(1+np.exp(-x))


def find_transformed_boundary_points(rf, grid, theta, image_dim):
    rf = rf.astype(np.int32)
    box_start = grid[rf[1], rf[0]]
    box_end = grid[rf[3], rf[2]]
    #batch_size = theta.shape[0]
    #box_start = box_start.unsqueeze(0).repeat(batch_size, 1)
    #box_end = box_end.unsqueeze(0).repeat(batch_size, 1)
    box_start = torch.cat([box_start, torch.ones(1).cuda()], dim=0).reshape(1, 3)
    box_end = torch.cat([box_end, torch.ones(1).cuda()], dim=0).reshape(1, 3)
    min_points = torch.mm(box_start, torch.t(theta)).view(-1)
    max_points = torch.mm(box_end, torch.t(theta)).view(-1)
    max_points = ((max_points + 1.) * image_dim) * 0.5
    min_points = ((min_points + 1.) * image_dim) * 0.5
    #print(min_points.cpu().numpy(), max_points.cpu().numpy())
    boundary_points = min_max_to_boundary_points(max_points.cpu().numpy(), min_points.cpu().numpy())
    return boundary_points


def min_max_to_boundary_points(min_points, max_points):
    boundary_points = np.zeros((5,2))
    boundary_points[0] = min_points
    boundary_points[1] = [min_points[0], max_points[1]]
    boundary_points[2] = max_points
    boundary_points[3] = [max_points[0], min_points[1]]
    boundary_points[4] = min_points
    return boundary_points


def plot_bboxes_of_few_images(imdb, detection, detection_NT, selected_indices, imagepath, epoch, gt_roidb, dir=None, image_res=320):
    """
    plots the ground truth, predicted box and no transform box of images of selected indices
    """
    if dir=="corLoc" and epoch==1:
        files = glob.glob(osp.join(imdb._devkit_path, dir) + '/*')
        for f in files:
            os.remove(f)
    result_path = osp.join(imdb._devkit_path, dir) + '/'

    for index in selected_indices:
        roidb = gt_roidb[index]
        filename = roidb['filename']
        gt_class = roidb['gt_classes'][0]
        class_name = imdb._classes[gt_class]
        gz_path = osp.join(imagepath, class_name, filename + '.JPEG.gz')
        with gzip.GzipFile(gz_path, 'r') as f:
            rgb_image = Image.open(f).resize((image_res, image_res), Image.BILINEAR).convert("RGB")
        plt.imshow(rgb_image)
        box = detection[index][2:].cpu().numpy()
        box_NT = detection_NT[index][2:].cpu().numpy()
        #class_index = int(detection[index][1].item())
        #print(class_index)
        #label = imdb.classes[class_index]
        min_points = [box[0], box[1]]
        max_points = [box[2], box[3]]
        #box_weight = sigmoid(box[4])
        boundary_points = min_max_to_boundary_points(min_points, max_points)
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], "r-", lw=1.5)
        #plt.text(boundary_points[0, 0]+1, boundary_points[0, 1], str(np.around(box_weight, 4)), color='green')
        min_points = [box_NT[0], box_NT[1]]
        max_points = [box_NT[2], box_NT[3]]
        boundary_points = min_max_to_boundary_points(min_points, max_points)
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], "b-", lw=1.5)

        boxes = roidb['boxes']
        for i in range(boxes.shape[0]):
            ground_truth = boxes[i]
            min_points = [ground_truth[0], ground_truth[1]]
            max_points = [ground_truth[2], ground_truth[3]]
            boundary_points = min_max_to_boundary_points(min_points, max_points)
            plt.plot(boundary_points[:, 0], boundary_points[:, 1], "#3BFF33", linestyle='-', lw=1.5)
        plt.savefig(result_path + "transform_" + str(epoch) + "_" + filename + ".png")
        plt.clf()
        plt.close()


