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

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp



def apply_transform(theta, grid):
    h, w, _ = grid.size()
    pos = grid.contiguous().view(-1, 2)
    pos = torch.cat([pos, torch.ones(h*w, 1).cuda()], dim=1)
    sampling_pos = torch.mm(pos, torch.t(theta))
    sampling_pos = sampling_pos.view(h, w, 2)
    return sampling_pos


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


def plot_bboxes_of_few_images_cub(imdb, detection, box_pos, selected_indices, imagepath, epoch, flag="train", dir=None, indicator=None):
    image_index = imdb._image_index
    image_names = imdb._image_names

    if flag in ("test", "corloc"):
        cache_dir = os.path.join(imdb._devkit_path, 'annotations_cache')
        cachefile = os.path.join(cache_dir, 'test_annots.pkl')
        with open(cachefile, "rb") as f:
            recs = pickle.load(f)
        keys = recs.keys()

    if dir=="corLoc":
        #files = glob.glob("../results/corLoc/*")
        files = glob.glob(osp.join(imdb._data_path, dir)+'/*')
        for f in files:
            os.remove(f)
    result_path = osp.join(imdb._data_path, dir)+'/'

    for index in selected_indices:
        image = image_index[index]
        image_at_path = osp.join(imagepath, image_names[index])
        rgb_image = Image.open(image_at_path).resize((320, 320), Image.BILINEAR).convert("RGB")
        plt.imshow(rgb_image)
        boxes = detection[index]['boxes']
        class_index = int(detection[index]['pred'].item())
        #print(class_index)
        #label = imdb.classes[class_index]
        max_pos = np.argmax(boxes[:, 4])
        count = 0
        for box in boxes:
            min_points = [box[0], box[1]]
            max_points = [box[2], box[3]]
            #box_weight = sigmoid(box[4])
            boundary_points = min_max_to_boundary_points(min_points, max_points)
            if count==max_pos:
                plt.plot(boundary_points[:, 0], boundary_points[:, 1], "b-", lw=1.5)
            else:
                plt.plot(boundary_points[:, 0], boundary_points[:, 1], "r-", lw=1.5)
            count += 1
        #plt.text(boundary_points[0, 0]+1, boundary_points[0, 1], label+'_'+str(np.around(box_weight, 4)), color='green')
        #fmap = box_pos.shape[0]
        #idx_y, idx_x = np.int(box[4])/fmap, np.int(box[4])%fmap
        #box_selected = box_pos[idx_y, idx_x]
        #min_points = [box_selected[0], box_selected[1]]
        #max_points = [box_selected[2], box_selected[3]]
        #boundary_points = min_max_to_boundary_points(min_points, max_points)
        #plt.plot(boundary_points[:, 0], boundary_points[:, 1], "b-", lw=1.5)

        if flag=="train":
            plt.savefig("../results/single_image/train/transform_"+str(epoch)+'_'+image+".png")
        elif flag=="test":
            key_i = keys[indicator[index]]
            ground_truth = recs[key_i]
            min_points = [ground_truth[1], ground_truth[2]]
            max_points = [ground_truth[3], ground_truth[4]]
            boundary_points = min_max_to_boundary_points(min_points, max_points)
            plt.plot(boundary_points[:, 0], boundary_points[:, 1], "#3BFF33", linestyle='-', lw=1.5)
            plt.savefig("../results/single_image/test/transform_"+str(epoch)+'_'+image+".png")
        else:
            key_i = keys[indicator[index]]
            ground_truth = recs[key_i]
            min_points = [ground_truth[1], ground_truth[2]]
            max_points = [ground_truth[3], ground_truth[4]]
            boundary_points = min_max_to_boundary_points(min_points, max_points)
            plt.plot(boundary_points[:, 0], boundary_points[:, 1], "#3BFF33", linestyle='-', lw=1.5)
            plt.savefig(result_path + "transform_"+str(epoch)+"_"+image+".png")
        plt.clf()
        plt.close()
