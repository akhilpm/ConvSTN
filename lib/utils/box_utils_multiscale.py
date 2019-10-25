import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import data_utils
import logging
import sys
import matplotlib.pyplot as plt
from utils.nms import nms


def get_transformed_bounding_box_cub(box_pos, theta, likelihood, batch_size, grid_position, H, x):
    bboxes = torch.zeros(batch_size, 5)
    count = 0
    pred = torch.argmax(likelihood, dim=1).view(-1, 1)

    #detection_scores = x.sum(1).detach()
    #num_classes = likelihood.shape[1]
    fmaps = [5, 10]
    scale = 0
    #no_transform = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).reshape(2,3).cuda()

    for k in range(batch_size):
        pred_class = pred[k].item()
        max_5map, max_10map = torch.max(x[0][k, pred_class]).item(), torch.max(x[1][k, pred_class]).item()
        #print(max_5map, max_10map)
        selected_map = np.argmax(np.array([max_5map, max_10map]))
        fmap = fmaps[selected_map]
        #print(k, pred_class)
        #print(x[selected_map].shape)
        detection_scores = x[selected_map][k, pred_class]
        max_of_k = torch.argmax(detection_scores).item()
        #max_of_k = torch.argmax(x_flat[k]).item()%(h*w)
        pos_y, pos_x = max_of_k/fmap, max_of_k%fmap
        rf = box_pos[selected_map][pos_y, pos_x]
        theta_ind = k*(fmap**2) + max_of_k
        max_points, min_points = find_transformed_boundary_points_cub(rf, grid_position, theta[selected_map][theta_ind], H)
        #max_points, min_points = find_transformed_boundary_points_cub(rf, grid_position, no_transform, H)
        bboxes[k, 0:2] =  torch.clamp(min_points, 0, H-1) #assuming H=W
        bboxes[k, 2:4] =  torch.clamp(max_points, 0, H-1)
        bboxes[k, 4] = max_of_k  #torch.max(detection_scores[k])
        scale += selected_map

    return bboxes, scale


def winner_scale(box_pos, theta, batch_size, grid_position, H, x):
    bboxes = torch.zeros(batch_size, 5)
    count = 0
    num_classes = x[0].shape[1]
    fmaps = [5, 10]
    scale = 0
    likelihood = torch.zeros(batch_size, num_classes).cuda()

    for k in range(batch_size):
        max_acts = torch.tensor([torch.max(x[0][k]), torch.max(x[1][k])])
        selected_map = torch.argmax(max_acts).item()
        fmap = fmaps[selected_map]
        class_scores = x[selected_map][k].contiguous().view(-1)
        class_scores = F.softmax(class_scores, dim=0).view(num_classes, fmap, fmap)
        probs = class_scores.view(num_classes, -1).sum(1)
        pred_class = torch.argmax(probs).item()
        likelihood[k] = probs
        detection_scores = x[selected_map][k, pred_class]
        max_of_k = torch.argmax(detection_scores).item()

        pos_y, pos_x = max_of_k/fmap, max_of_k%fmap
        rf = box_pos[selected_map][pos_y, pos_x]
        theta_ind = k*(fmap**2) + max_of_k
        max_points, min_points = find_transformed_boundary_points_cub(rf, grid_position, theta[selected_map][theta_ind], H)
        #max_points, min_points = find_transformed_boundary_points_cub(rf, grid_position, no_transform, H)
        bboxes[k, 0:2] =  torch.clamp(min_points, 0, H-1) #assuming H=W
        bboxes[k, 2:4] =  torch.clamp(max_points, 0, H-1)
        bboxes[k, 4] = max_of_k  #torch.max(detection_scores[k])
        scale += selected_map

    return bboxes, scale, likelihood


def min_max_to_boundary_points(min_points, max_points):
    boundary_points = np.zeros((5,2))
    boundary_points[0] = min_points
    boundary_points[1] = [min_points[0], max_points[1]]
    boundary_points[2] = max_points
    boundary_points[3] = [max_points[0], min_points[1]]
    boundary_points[4] = min_points
    return boundary_points


def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def plot_all_boxes(box_pos, theta, grid_position, x, rgb_image, fmap, H, epoch):
    detection_scores = x.sum(0)
    max_score = torch.max(detection_scores).item()
    no_transform = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).reshape(2,3).cuda()
    for i in range(fmap**2):
        plt.imshow(rgb_image)
        pos_y, pos_x  = i/fmap, i%fmap
        #print(pos_y, pos_x)
        rf = box_pos[pos_y, pos_x]

        #plot transformed RF
        max_points, min_points = find_transformed_boundary_points_cub(rf, grid_position, theta[i], H)
        min_points, max_points = torch.clamp(min_points, 0, H-1), torch.clamp(max_points, 0, H-1)
        boundary_points = min_max_to_boundary_points(min_points.cpu().numpy(), max_points.cpu().numpy())
        box_weight = sigmoid(detection_scores[pos_y, pos_x].cpu().numpy())
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], "r-", lw=1.5)

        if max_score == detection_scores[pos_y, pos_x]:
            plt.text(boundary_points[0, 0]+1, boundary_points[0, 1], 'MAX_'+str(np.around(box_weight, 4)), color='red')
        else:
            plt.text(boundary_points[0, 0]+1, boundary_points[0, 1], str(np.around(box_weight, 4)), color='red')

        #plot RF
        max_points, min_points = find_transformed_boundary_points_cub(rf, grid_position, no_transform, H)
        min_points, max_points = torch.clamp(min_points, 0, H-1), torch.clamp(max_points, 0, H-1)
        boundary_points = min_max_to_boundary_points(min_points.cpu().numpy(), max_points.cpu().numpy())
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], "b-", lw=1.5)

        plt.savefig("../results/all_bboxes/pos_" + str(epoch) + '_' + str(pos_y+1) + '_'+ str(pos_x+1) + ".png")
        plt.clf()
        plt.close()



def find_transformed_boundary_points_cub(rf, grid, theta, image_dim):
    rf = rf.astype(np.int32)
    box_start = grid[rf[1], rf[0]]
    box_end = grid[rf[3], rf[2]]
    #batch_size = theta.shape[0]
    #print(batch_size, theta)
    #box_start = box_start.unsqueeze(0).repeat(batch_size, 1)
    #box_end = box_end.unsqueeze(0).repeat(batch_size, 1)
    #print(box_start, box_end)
    box_start = torch.cat([box_start, torch.ones(1).cuda()], dim=0)
    box_end = torch.cat([box_end, torch.ones(1).cuda()], dim=0)
    min_points = torch.mm(box_start.view(1, -1), theta.permute(1, 0))
    max_points = torch.mm(box_end.view(1, -1), theta.permute(1, 0))
    max_points = ((max_points + 1.) * image_dim) * 0.5
    min_points = ((min_points + 1.) * image_dim) * 0.5
    return max_points[0], min_points[0]



def apply_transform(theta, grid):
    h, w, _ = grid.size()
    pos = grid.contiguous().view(-1, 2)
    pos = torch.cat([pos, torch.ones(h*w, 1).cuda()], dim=1)
    sampling_pos = torch.mm(pos, torch.t(theta))
    sampling_pos = sampling_pos.view(h, w, 2)
    return sampling_pos


def get_detected_boxes_cub(all_boxes, x, detection, likelihood, start_indx, batch_size):
    """ select the top scoring bounding box at each position, assign it to the corresponding class and apply NMS """
    detection_scores = x.sum(1)
    h, w = x.shape[2], x.shape[3]
    #print(max_at_each_pos.shape)
    num_classes = x.shape[1]

    for k in range(batch_size):
        max_of_k = torch.argmax(likelihood[k]).item()
        image_ind = k+start_indx
        detection[max_of_k][image_ind] = torch.cat([detection[max_of_k][image_ind], all_boxes[k].view(1, -1)], dim=0)

    return detection
