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



def find_receptive_field(image, grid, y_low, y_high, x_low, x_high, num_layers_above, H, W):
#def find_receptive_field(image, grid, x_low, x_high, y_low, y_high, num_layers_above, H, W):

    multiplier = np.power(2, num_layers_above)
    rf_start_x = min(W, max(0, multiplier*x_low))
    rf_end_x = min(W, max(0, multiplier*(x_high+1)))
    rf_start_y = min(H, max(0, multiplier*y_low))
    rf_end_y = min(H, max(0, multiplier*(y_high+1)))
    rf = image[:, rf_start_y:rf_end_y, rf_start_x:rf_end_x]
    grid_pos = grid[rf_start_y:rf_end_y, rf_start_x:rf_end_x, :]
    box_cord = torch.tensor([[rf_start_x, rf_start_y, 1], [rf_start_x, rf_end_y, 1],
        [rf_end_x, rf_end_y, 1], [rf_end_x, rf_start_y, 1], [rf_start_x, rf_start_y, 1]], dtype=torch.float).cuda()
    #print(rf_start_x, rf_end_x, rf_start_y, rf_end_y)
    return rf, grid_pos, box_cord



def apply_transform(theta, grid):
    h, w, _ = grid.size()
    pos = grid.contiguous().view(-1, 2)
    pos = torch.cat([pos, torch.ones(h*w, 1).cuda()], dim=1)
    sampling_pos = torch.mm(pos, torch.t(theta))
    sampling_pos = sampling_pos.view(h, w, 2)
    return sampling_pos


def plot_receptive_field_transform(theta, data, grid_positions, H, W, epoch, fmap_d):
    """ Plot all transforms on all figures. It is hard to interpret this way """
    fig, ax = plt.subplots(2, 2)
    count = 0
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g']
    for m in range(4):

        image = convert_image_np(data[m].cpu())
        loc_0, loc_1 = m/2, m%2
        ax[loc_0, loc_1].imshow(image)

        for i in range(0, fmap_d):
            for j in range(0, fmap_d):
                rf, grid_pos, _ = find_receptive_field(data[m], grid_positions[m], i-1, i+1, j-1, j+1, 3, H, W)

                #expand dimension of rf and theta for grid generating function
                #rf = rf.unsqueeze(0)
                #theta_loc = theta[count].unsqueeze(0)
                #print(rf.size())
                #grid = F.affine_grid(theta_loc, rf.size())
                grid = apply_transform(theta[count], grid_pos)
                #print(grid.shape)

                #convert normalized coordinates back to original image coordinates
                grid_x, grid_y = grid[:, :, 1], grid[:, :, 0]
                grid_x = ((grid_x + 1.) * W) * 0.5
                grid_y = ((grid_y + 1.) * H) * 0.5
                
                #clip the coordinates to image height and width
                grid_x =  torch.clamp(grid_x, 0, W-1)
                grid_y =  torch.clamp(grid_y, 0, H-1)

                grid_x_flat = grid_x.view(grid_x.numel()).cpu().numpy()
                grid_y_flat = grid_y.view(grid_y.numel()).cpu().numpy()              

                points = np.stack([grid_x_flat, grid_y_flat], axis=1)
                #print(points.shape)
                hull = ConvexHull(points)
                boundary_points = np.append(hull.vertices, hull.vertices[0])
                #print("{}||||||{} " .format(points[boundary_points,0], points[boundary_points,1]))
                ax[loc_0, loc_1].plot(points[boundary_points,0], points[boundary_points,1], colors[i*4+j]+'-', lw=1.0)

                count += 1
        print("="*30)        

    plt.savefig("results/local/transform_"+str(epoch)+".png")
    plt.clf()
    plt.close(fig)


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


def plot_receptive_field_transform_single_image(box_pos, theta, data, grid, H, W, epoch, batch_id, fmap, log, det_scores, flag, img_indx):
    """ Each transform on the same image is plotted as separate figure """
    start_indx = img_indx*(fmap**2)
    count = 0
    image = convert_image_np(data.cpu())
    plt.imshow(image)

    for i in range(fmap):
        for j in range(fmap):
            rf = box_pos[i, j]
            #compute the transformed corner points
            boundary_points = find_transformed_boundary_points(rf, grid, theta[start_indx+count], H)
            #boundary_points = torch.mm(box_cord, torch.t(theta[start_indx+count]))
            box_weight = det_scores[img_indx, i, j]
            #clip the coordinates to image height and width
            grid_x =  np.clip(boundary_points[:, 0], 0, W-1)
            grid_y =  np.clip(boundary_points[:, 1], 0, H-1)
            #log.info("Epoch: {}, pos: {}, {}, x_cord:{}".format(epoch, i, j, grid_x))
            #log.info("Epoch: {}, pos: {}, {}, y_cord:{}".format(epoch, i, j, grid_y))
            plt.plot(grid_x, grid_y, 'r-', lw=10*box_weight)
            count += 1
    if flag=="train":
        plt.savefig("../results/single_image/train/transform_"+str(epoch)+'_'+str(batch_id)+".png")
    else:
        plt.savefig("../results/single_image/test/transform_"+str(epoch)+'_'+str(batch_id)+".png")
    plt.clf()
    plt.close()


def sigmoid(x):
    return 1.0/(1+np.exp(-x))


def plot_bboxes_of_few_images(imdb, detection, selected_indices, imagepath, epoch, flag="train"):
    for index in selected_indices:
        image = imdb._image_index[index]
        image_at_path = osp.join(imagepath, image+'.jpg')
        rgb_image = Image.open(image_at_path).resize((512, 512), Image.BILINEAR).convert("RGB")
        plt.imshow(rgb_image)
        for k in range(1, imdb.num_classes):
            boxes = detection[k][index].cpu().numpy()
            label = imdb.classes[k]
            if len(boxes)==0:
                continue
            for box in boxes:
                min_points = [box[0], box[1]]
                max_points = [box[2], box[3]]
                box_weight = sigmoid(box[4])
                boundary_points = min_max_to_boundary_points(min_points, max_points)
                plt.plot(boundary_points[:, 0], boundary_points[:, 1], "r-", lw=box_weight)
                plt.text(boundary_points[0, 0]+1, boundary_points[0, 1], label+'_'+str(np.around(box[4], 4)), color='green')
        if flag=="train":
            plt.savefig("../results/single_image/train/transform_"+str(epoch)+'_'+image+".png")
        else:
            plt.savefig("../results/single_image/test/transform_"+str(epoch)+'_'+image+".png")
        plt.clf()
        plt.close()


def plot_bboxes_of_few_images_cub(imdb, detection, selected_indices, imagepath, epoch, flag="train", dir=None):
    image_index = imdb._image_index
    image_names = imdb._image_names

    if dir=="corLoc":
        files = glob.glob("../results/corLoc/*")
        for f in files:
            os.remove(f)

    for index in selected_indices:
        image = image_index[index]
        image_at_path = osp.join(imagepath, image_names[index])
        rgb_image = Image.open(image_at_path).resize((320, 320), Image.BILINEAR).convert("RGB")
        plt.imshow(rgb_image)
        box = detection[index][2:].cpu().numpy()
        class_index = int(detection[index][1].item())
        #print(class_index)
        label = imdb.classes[class_index]
        min_points = [box[0], box[1]]
        max_points = [box[2], box[3]]
        box_weight = sigmoid(box[4])
        boundary_points = min_max_to_boundary_points(min_points, max_points)
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], "r-", lw=1.5)
        plt.text(boundary_points[0, 0]+1, boundary_points[0, 1], label+'_'+str(np.around(box_weight, 4)), color='green')
        if flag=="train":
            plt.savefig("../results/single_image/train/transform_"+str(epoch)+'_'+image+".png")
        elif flag=="test":
            plt.savefig("../results/single_image/test/transform_"+str(epoch)+'_'+image+".png")
        else:
            plt.savefig("../results/corLoc/transform_"+str(epoch)+"_"+image+".png")
        plt.clf()
        plt.close()
