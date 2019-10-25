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


def find_iou(BBGT, bb):
    ixmin = np.maximum(BBGT[0], bb[0])
    iymin = np.maximum(BBGT[1], bb[1])
    ixmax = np.minimum(BBGT[2], bb[2])
    iymax = np.minimum(BBGT[3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
            (BBGT[2] - BBGT[0] + 1.) *
            (BBGT[3] - BBGT[1] + 1.) - inters)
    overlap = inters / uni
    return overlap

def get_area(minp, maxp):
    return (maxp[0]-minp[0]+1.0) * (maxp[1]-minp[1]+1.0)


def get_transformed_bounding_box_cub(box_pos, theta, likelihood, batch_size, grid_position, H, x):
    #batch_size = len(data)
    #no_transform = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).reshape(1,2,3).cuda()
    #no_transform = no_transform.repeat(batch_size, 1, 1)
    #grid_position = F.affine_grid(no_transform, data[:1].size())
    bboxes = {k:np.zeros((0, 5)) for k in range(batch_size)}
    count = 0
    pred = torch.argmax(likelihood, dim=1).view(-1, 1)

    #detection_scores = x.sum(1)
    h, w = x.shape[2], x.shape[3]
    fmap = h
    #example_boxes = torch.zeros(0, 5)
    #print(max_at_each_pos.shape)
    num_classes = likelihood.shape[1]
    no_transform = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).reshape(2,3).cuda()

    for k in range(batch_size):
        pred_class = pred[k]
        detection_scores = x[k, pred_class].squeeze(0)
        #print(detection_scores.shape)
        for idx_y in range(fmap):
            for idx_x in range(fmap):
                rf = box_pos[idx_y, idx_x]
                max_points, min_points = find_transformed_boundary_points_cub(rf, grid_position, theta[count], H)
                #area_before_clamp = get_area(min_points, max_points)
                #max_points, min_points = find_transformed_boundary_points_cub(rf, grid_position, no_transform, H)
                min_points, max_points = torch.clamp(min_points, 0, H-1), torch.clamp(max_points, 0, H-1)
                #area_after_clamp = get_area(min_points, max_points)
                #if (area_after_clamp / area_before_clamp) >= 0.3:
                min_points, max_points = min_points.cpu().detach().numpy(), max_points.cpu().detach().numpy()
                transformed_box = np.array([min_points[0], min_points[1], max_points[0], max_points[1], detection_scores[idx_y, idx_x]])
                bboxes[k] = np.append(bboxes[k], transformed_box.reshape(1, -1), axis=0)
                count += 1


        #pred_class = pred[k]
        #detection_scores = x[k, pred_class]

        if len(bboxes[k])==0:
            bboxes[k] = np.zeros((1, 5))
            max_of_k = torch.argmax(detection_scores).item()
            pos_y, pos_x = max_of_k/h, max_of_k%h
            rf = box_pos[pos_y, pos_x]
            theta_ind = k*(fmap**2) + max_of_k
            max_points, min_points = find_transformed_boundary_points_cub(rf, grid_position, theta[theta_ind], H)
            bboxes[k][0, 0:2] =  torch.clamp(min_points, 0, H-1).cpu().numpy() #assuming H=W
            bboxes[k][0, 2:4] =  torch.clamp(max_points, 0, H-1).cpu().numpy()
            bboxes[k][0, 4] = torch.max(detection_scores)

    #rf = box_pos_32[pos_y, pos_x]
    #max_points, min_points = find_transformed_boundary_points_cub(rf, grid_position, theta[theta_ind], H)
    #min_points, max_points = torch.clamp(min_points, 0, H-1), torch.clamp(max_points, 0, H-1)
    #min_points, max_points = min_points.cpu().detach().numpy(), max_points.cpu().detach().numpy()
    #transformed_box = np.array([min_points[0], min_points[1], max_points[0], max_points[1], max_of_k])
    #bboxes[k] = np.append(bboxes[k], transformed_box.reshape(1, -1), axis=0)

    return bboxes


def plot_all_boxes(box_pos, theta, grid_position, x, rgb_image, fmap, H, epoch, flag):
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
        boundary_points = min_max_to_boundary_points(min_points, max_points)
        box_weight = sigmoid(detection_scores[pos_y, pos_x]).item()
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], "r-", lw=1.5)

        if max_score == detection_scores[pos_y, pos_x]:
            plt.text(boundary_points[0, 0]+1, boundary_points[0, 1], 'MAX_'+str(np.around(box_weight, 4)), color='red')
        else:
            plt.text(boundary_points[0, 0]+1, boundary_points[0, 1], str(np.around(box_weight, 4)), color='red')

        #plot RF
        max_points, min_points = find_transformed_boundary_points_cub(rf, grid_position, no_transform, H)
        min_points, max_points = torch.clamp(min_points, 0, H-1), torch.clamp(max_points, 0, H-1)
        boundary_points = min_max_to_boundary_points(min_points, max_points)
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], "b-", lw=1.5)

        if flag=="train":
            plt.savefig("../results/all_bboxes/train/pos_" + str(epoch) + '_' + str(pos_y+1) + '_'+ str(pos_x+1) + ".png")
        else:
            plt.savefig("../results/all_bboxes/test/pos_" + str(epoch) + '_' + str(pos_y+1) + '_'+ str(pos_x+1) + ".png")
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
