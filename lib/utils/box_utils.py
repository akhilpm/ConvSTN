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


def get_bb_after_transform(theta, data, H, W, epoch, fmap):

    batch_size = len(data)
    no_transform = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).reshape(1,2,3).cuda()
    #no_transform = no_transform.repeat(batch_size, 1, 1)
    grid_position = F.affine_grid(no_transform, data[:1].size())
    bboxes = torch.zeros(batch_size, fmap**2, 5)

    for i in range(fmap**2):
        loc_0, loc_1 = i/fmap, i%fmap
        grid_pos, box_cord = find_receptive_field(grid_position[0], loc_0-1, loc_0+1, loc_1-1, loc_1+1, 3, H, W)
        #print(grid_pos.shape)
        max_points, min_points = find_transformed_boundary_points(grid_pos, theta[i::fmap**2], H)
        #print("boundary points computed")
        #print(boundary_points)
        bboxes[:, i, 0] =  torch.clamp(min_points[:, 0], 0, W-1)
        bboxes[:, i, 1] =  torch.clamp(min_points[:, 1], 0, H-1)
        bboxes[:, i, 2] =  torch.clamp(max_points[:, 0], 0, W-1)
        bboxes[:, i, 3] =  torch.clamp(max_points[:, 1], 0, H-1)
        #bboxes[:, i, 4] = det_scores[:, loc_0, loc_1]

    return bboxes


def get_transformed_bounding_box(box_pos, theta, data, grid_position, H, W, epoch, fmap):

    batch_size = len(data)
    #no_transform = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).reshape(1,2,3).cuda()
    #no_transform = no_transform.repeat(batch_size, 1, 1)
    #grid_position = F.affine_grid(no_transform, data[:1].size())
    bboxes = torch.zeros(batch_size, fmap**2, 5)
    count = 0

    for i in range(fmap):
        for j in range(fmap):
            rf = box_pos[i, j]
            #print(grid_pos.shape)
            max_points, min_points = find_transformed_boundary_points(rf, grid_position, theta[count::fmap**2], H)
            #print(min_points, max_points)
            #print("boundary points computed")
            #print(boundary_points)
            bboxes[:, count, 0] =  torch.clamp(min_points[:, 0], 0, W-1)
            bboxes[:, count, 1] =  torch.clamp(min_points[:, 1], 0, H-1)
            bboxes[:, count, 2] =  torch.clamp(max_points[:, 0], 0, W-1)
            bboxes[:, count, 3] =  torch.clamp(max_points[:, 1], 0, H-1)
            #bboxes[:, i, 4] = det_scores[:, loc_0, loc_1]
            count+=1 

    return bboxes


def get_transformed_bounding_box_cub(box_pos, theta, likelihood, batch_size, grid_position, H, W, epoch, fmap, x):
    #batch_size = len(data)
    #no_transform = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).reshape(1,2,3).cuda()
    #no_transform = no_transform.repeat(batch_size, 1, 1)
    #grid_position = F.affine_grid(no_transform, data[:1].size())
    bboxes = torch.zeros(batch_size, 5)
    count = 0
    #pred = torch.argmax(likelihood, dim=1).view(-1, 1)

    detection_scores = x.sum(1).detach()
    h, w = x.shape[2], x.shape[3]
    #example_boxes = torch.zeros(0, 5)
    #print(max_at_each_pos.shape)
    num_classes = likelihood.shape[1]
    no_transform = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).reshape(2,3).cuda()

    for k in range(batch_size):
        #pred_class = pred[k]
        #detection_scores = x[k, pred_class]
        max_of_k = torch.argmax(detection_scores[k]).item()
        pos_y, pos_x = max_of_k/h, max_of_k%h
        rf = box_pos[pos_y, pos_x]
        theta_ind = k*(fmap**2) + max_of_k
        max_points, min_points = find_transformed_boundary_points_cub(rf, grid_position, theta[theta_ind], H)
        #max_points, min_points = find_transformed_boundary_points_cub(rf, grid_position, no_transform, H)
        bboxes[k, 0:2] =  torch.clamp(min_points, 0, H-1) #assuming H=W
        bboxes[k, 2:4] =  torch.clamp(max_points, 0, H-1)
        bboxes[k, 4] = max_of_k  #torch.max(detection_scores[k])

    return bboxes



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



def find_transformed_boundary_points(rf, grid, theta, image_dim):
    rf = rf.astype(np.int32)
    box_start = grid[rf[1], rf[0]]
    box_end = grid[rf[3], rf[2]]
    batch_size = theta.shape[0]
    #print(batch_size, theta)
    box_start = box_start.unsqueeze(0).repeat(batch_size, 1)
    box_end = box_end.unsqueeze(0).repeat(batch_size, 1)
    #print(box_start, box_end)
    box_start = torch.cat([box_start, torch.ones(batch_size, 1).cuda()], dim=1)
    box_end = torch.cat([box_end, torch.ones(batch_size, 1).cuda()], dim=1)
    min_points = torch.bmm(box_start.view(batch_size, 1, 3), theta.permute(0,2,1)).squeeze(1)
    max_points = torch.bmm(box_end.view(batch_size, 1, 3), theta.permute(0,2,1)).squeeze(1)
    max_points = ((max_points + 1.) * image_dim) * 0.5
    min_points = ((min_points + 1.) * image_dim) * 0.5
    return max_points, min_points


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


def find_receptive_field(grid, y_low, y_high, x_low, x_high, num_layers_above, H, W):
#def find_receptive_field(image, grid, x_low, x_high, y_low, y_high, num_layers_above, H, W):

    multiplier = np.power(2, num_layers_above)
    rf_start_x = min(W, max(0, multiplier*x_low))
    rf_end_x = min(W, max(0, multiplier*(x_high+1)))
    rf_start_y = min(H, max(0, multiplier*y_low))
    rf_end_y = min(H, max(0, multiplier*(y_high+1)))
    grid_pos = grid[rf_start_y:rf_end_y, rf_start_x:rf_end_x, :]
    box_cord = torch.tensor([[rf_start_x, rf_start_y, 1], [rf_start_x, rf_end_y, 1],
        [rf_end_x, rf_end_y, 1], [rf_end_x, rf_start_y, 1], [rf_start_x, rf_start_y, 1]], dtype=torch.float).cuda()
    #print(rf_start_x, rf_end_x, rf_start_y, rf_end_y)
    return grid_pos, box_cord



def apply_transform(theta, grid):
    h, w, _ = grid.size()
    pos = grid.contiguous().view(-1, 2)
    pos = torch.cat([pos, torch.ones(h*w, 1).cuda()], dim=1)
    sampling_pos = torch.mm(pos, torch.t(theta))
    sampling_pos = sampling_pos.view(h, w, 2)
    return sampling_pos


def get_class_specific_boxes(selected_boxes, all_boxes, detection, pos_class_scores, class_indx, fmap, start_indx):
    """
    given all_boxes in each_image, obtain the selected boxes for each image for 
    the particular class indicated by class_index, append it's score.
    ---> all_boxes = [batch_size, fmap*fmap, 5]
    ---> selected_boxes =  [no_images, selected_indices] (list)
    ---> pos_class_scores = [batch_size, no_classes+1, fmap, fmap]
    ---> detection = {classes: {iamges_id:{box_cordinates and score of that each selected box} }} (dict)
    """
    count = 0
    for i in range(len(selected_boxes)):
        no_boxes = len(selected_boxes[i])
        boxes = selected_boxes[i]
        detection[class_indx][start_indx+i] = np.zeros((no_boxes, 5))
        for j in range(no_boxes):
            detection[class_indx][i+start_indx][j] = all_boxes[i, boxes[j]]
            pos_x, pos_y = boxes[j]%fmap, boxes[j]/fmap
            detection[class_indx][i+start_indx][j, 4] = pos_class_scores[i, class_indx, pos_y, pos_x]
        count+=1

    return detection



def get_detected_boxes(all_boxes, x, detection, fmap, start_indx, batch_size):
    """ select the top scoring bounding box at each position, assign it to the corresponding class and apply NMS """
    max_at_each_pos = torch.argmax(x, dim=1)
    example_boxes = torch.zeros(0, 5)
    #print(max_at_each_pos.shape)
    num_classes = x.shape[1]

    for i in range(fmap):
        for j in range(fmap):
            max_boxes = max_at_each_pos[:, i, j]
            for k in range(batch_size):
                max_of_k = int(max_boxes[k].cpu().numpy())
                #print(max_boxes, max_of_k, type(max_of_k))
                image_ind = k+start_indx
                if max_of_k==0:
                    continue
                #else:
                    #if len(detection[max_of_k][image_ind])==0:
                    #    detection[max_of_k][image_ind] = all_boxes[k, i*fmap+j].view(1, -1)
                    #else:
                if x[k, max_of_k, i, j]>=0.001:
                    all_boxes[k, i*fmap+j, 4] = x[k, max_of_k, i, j]
                    detection[max_of_k][image_ind] = torch.cat([detection[max_of_k][image_ind], all_boxes[k, i*fmap+j].view(1, -1)], dim=0)

    #apply NMS
    for i in range(1, num_classes):
        for j in range(batch_size):
            image_ind = j+start_indx
            if len(detection[i][image_ind])!=0:
                selected_boxes = nms(detection[i][image_ind], 0.4)
                no_boxes = len(selected_boxes)
                #print(i, j, no_boxes)
                total_boxes = detection[i][image_ind].clone()
                detection[i][image_ind] = torch.zeros(no_boxes, 5)
                for k in range(no_boxes):
                    detection[i][image_ind][k] = total_boxes[selected_boxes[k]]

    return detection


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
