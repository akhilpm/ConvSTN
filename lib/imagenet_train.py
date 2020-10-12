from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from itertools import compress
plt.ioff()
import numpy as np
import logging
import sys
import pickle
import argparse
import os
import os.path as osp
#from PIL import Image
from torch.optim.lr_scheduler import MultiStepLR
#from copy import deepcopy

from utils import data_utils
from utils import vis_utils_imagenet
#from model.model_CUB_multiscale import Net
from model.model_CUB_new import Net
#from model.config import get_output_dir
from datasets.image_net import image_net
from datasets import imagenet_eval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_acc1 = 0.0
best_loc_acc1 = 0.0
test_acc2 = 0.0
best_loc_acc2 = 0.0
test_im_index = 90
#fmap = 10 #for VGG
fmap = 5 #for ResNet
test_im_index = 1300
#selected_indices = [797, 1597, 2397, 3197, 3997, 4797]
#selected_indices2 = [10, 30, 60, 90]
selected_indices= [10, 100, 360, 450]
#selected_indices = [10, 30, 600, 900]
lambda_l2 = 0.001
lambda_reg_score = 5.0
class_train_acc = []
loc_train_acc_max = []
loc_train_acc_all = []
loc_train_acc_max_NT = []
loc_train_acc_all_NT = []

class_test_acc1 = []
loc_test_acc1_max = []
loc_test_acc1_all = []
loc_test_acc1_max_NT = []
loc_test_acc1_all_NT = []

class_test_acc2 = []
loc_test_acc2_max = []
loc_test_acc2_all = []
loc_test_acc2_max_NT = []
loc_test_acc2_all_NT = []


def hook_backward(module, grad_input, grad_output):
    #print('grad_input upto 10:', grad_input[0][:10])
    #print('grad_output upto 10:', grad_output[0][:10])
    #sing = np.linalg.svg(grad_input.numpy(), compute_uv=False)
    #print("singlular values: {}" .format(sing))
    #print('grad_input size:', len(grad_input))
    #print('grad_output size:', grad_output[0].size())
    #print('grad_input norm:', grad_input[0].norm())
    print(grad_input[0].shape, grad_input[1].shape, grad_input[2].shape)


def hook_forward(module, input, output):
    print(module.__class__.__name__ + ' forward')
    #print(len(input), len(output))
    first_image_out = output[:16].data.cpu().detach().numpy()
    first_image_in = input[0][:16].cpu().detach().numpy()
    log.info(np.linalg.norm(first_image_in, axis=1))
    log.info(np.linalg.norm(first_image_out, axis=1))


def train_model(model, imdb, log, optimizer, epoch, batch_size, transform, image_res):
    model.train()
    correct = 0
    count = 0
    train_loss = 0.0
    acc = 0.0
    #imdb._image_index = imdb._image_index[:100]
    #imdb._image_labels = imdb._image_labels[:100]
    #imdb.gt_roidb = imdb.gt_roidb[:1000]
    train_size = len(imdb.gt_roidb)
    imagepath = osp.join(imdb._data_path, "train")

    index_shuf = np.random.permutation(train_size)
    #print(index_shuf)
    gt_roidb = [imdb.gt_roidb[i] for i in index_shuf]
    #index_shuf = [int(imdb._image_index[index])-1 for index in index_shuf]

    detection_max = torch.zeros(train_size, 7) #(columns: image_index(1), class_index(1), box_cords(4), conf_score(1))
    detection_max[:, 0] = torch.tensor(index_shuf)

    detection_max_NT = torch.zeros(train_size, 7) #(columns: image_index(1), class_index(1), box_cords(4), conf_score(1))
    detection_max_NT[:, 0] = torch.tensor(index_shuf)

    if train_size%batch_size==0:
        num_batches_per_epoch = int(train_size/batch_size)
    else:
        num_batches_per_epoch = int(train_size/batch_size) + 1
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num+1)*batch_size, train_size)
        trainX, trainY = data_utils.load_imagenet_data(imagepath, imdb, gt_roidb, start_index, end_index, transform, image_res, "train")
        data, target = trainX.to(device), trainY.to(device)
        #print(target.shape)

        optimizer.zero_grad()
        count += 1
        #likelihood, all_boxes, theta_diff, reg_score = model(data, image_res, epoch, log, batch_id=count, batch_size=data.shape[0])
        likelihood, x, all_boxes, theta_diff = model(data, image_res, epoch, log, batch_id=count, batch_size=data.shape[0])
        #print(likelihood.shape)
        #likelihood = torch.clamp(likelihood, min=0.0)

        #loss = torch.sum(torch.log(target*(likelihood - 0.5) + 0.5))
        loss1 = torch.sum(-target*torch.log(likelihood))/likelihood.shape[0]
        print("Loss CE {}".format(loss1))
        #print(torch.max(likelihood, dim=1))
        #log.info("Score reg loss: {}".format(lambda_reg_score * reg_score))
        loss = loss1 + lambda_l2 * theta_diff.pow(2).sum() # + lambda_reg_score * reg_score #+ loss2 #+ loss3
        #print("Total Loss {}".format(loss))
        #print(torch.sum(torch.argmax(likelihood, dim=1)==torch.argmax(target, dim=1)))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*likelihood.shape[0]
        if batch_num % 200 == 0:
            log.info("[TRAIN] Epoch: {} batch: {} loss: {}".format(epoch, count, loss.item()))

        with torch.no_grad():
            pred = torch.argmax(likelihood, dim=1).view(-1, 1)
            batch_acc = torch.sum(pred == torch.argmax(target, dim=1).view(-1, 1)).float().cpu().numpy()
            acc += batch_acc
            detection_max[start_index:end_index, 1] = pred.view(-1)
            detection_max[start_index:end_index, 2:] = all_boxes[0]
            detection_max_NT[start_index:end_index, 1] = pred.view(-1)
            detection_max_NT[start_index:end_index, 2:] = all_boxes[1]


    #with open("../results/detection.pkl", 'wb') as fid:
    #    pickle.dump(detection, fid, pickle.HIGHEST_PROTOCOL)

    with torch.no_grad():
        #cache_file = osp.join(imdb._devkit_path, 'cache', imdb.name + '_' + imdb._image_set + '_gt_roidb.pkl')
        #cache_file = osp.join(imdb._devkit_path, 'cache', imdb.name + '_' + imdb._image_set + '_' + str(image_res) + '_gt_roidb.pkl')
        train_loss /= train_size
        acc =  acc / train_size
        log.info("[TRAIN] loss: {} Accuracy: {}".format(train_loss, acc))
        class_train_acc.append(acc)

        loca_acc, corLoc = imagenet_eval.eval(detection_max, gt_roidb, log)
        loc_train_acc_max.append(loca_acc)

        loca_acc, corLoc = imagenet_eval.eval(detection_max_NT, gt_roidb, log)
        loc_train_acc_max_NT.append(loca_acc)
        vis_utils_imagenet.plot_bboxes_of_few_images(imdb, detection_max, detection_max_NT, selected_indices, imagepath, epoch, gt_roidb, "corLoc", image_res)



def test_model(model, imdb, log, epoch, batch_size, area_hist, transform, image_res):
    global test_acc1, test_acc2
    global best_loc_acc1, best_loc_acc2
    global test_im_index

    #imdb._image_index = imdb._image_index[:100]
    #imdb._image_labels = imdb._image_labels[:100]

    imagepath = osp.join(imdb._data_path, "validation")
    with torch.no_grad():
        model.eval()
        test_loss = 0.0
        acc = 0.0
        correct = 0
        count = 0
        #imdb.gt_roidb = imdb.gt_roidb[:1000]
        test_size = len(imdb.gt_roidb)

        if test_size%batch_size==0:
            num_batches_per_epoch = int(test_size/batch_size)
        else:
            num_batches_per_epoch = int(test_size/batch_size) + 1

        detection_max = torch.zeros(test_size, 7) #(columns: image_index, class_index, box_cords, conf_score)
        detection_max[:, 0] = torch.tensor(np.arange(test_size))

        detection_max_NT = torch.zeros(test_size, 7) #(columns: image_index, class_index, box_cords, conf_score)
        detection_max_NT[:, 0] = torch.tensor(np.arange(test_size))

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num+1)*batch_size, test_size)
            testX, testY = data_utils.load_imagenet_data(imagepath, imdb, imdb.gt_roidb, start_index, end_index, transform, image_res, "test")
            data, target = testX.to(device), testY.to(device)

            count += 1
            #likelihood, all_boxes, theta_diff, reg_score = model(data, image_res, epoch, log, batch_id=count, batch_size=data.shape[0])
            likelihood, x, all_boxes, theta_diff = model(data, image_res, epoch, log, batch_id=count, batch_size=data.shape[0])


            loss1 = torch.sum(-target*torch.log(likelihood))/likelihood.shape[0]
            loss = loss1 + lambda_l2 * theta_diff.pow(2).sum() #+ lambda_reg_score * reg_score #+ loss2 + loss3

            test_loss += loss.item()*likelihood.shape[0]
            if batch_num % 100 == 0:
                log.info("[TEST] Epoch: {} batch: {} loss: {}".format(epoch, count, loss.item()))

            pred = torch.argmax(likelihood, dim=1).view(-1, 1)
            batch_acc = torch.sum(pred == torch.argmax(target, dim=1).view(-1, 1)).float().cpu().numpy()
            acc += batch_acc

            detection_max[start_index:end_index, 1] = pred.view(-1)
            detection_max[start_index:end_index, 2:] = all_boxes[0]
            detection_max_NT[start_index:end_index, 1] = pred.view(-1)
            detection_max_NT[start_index:end_index, 2:] = all_boxes[1]


        test_loss /= test_size
        acc =  acc / test_size
        log.info("[TEST] loss:{} Accuracy: {}".format(test_loss, acc))

        #cache_file = osp.join(imdb._devkit_path, 'cache', imdb.name + '_' + imdb._image_set + '_' + str(image_res) + '_gt_roidb.pkl')
        class_test_acc1.append(acc)
        loca_acc, corLoc = imagenet_eval.eval(detection_max, imdb.gt_roidb, log)
        loc_test_acc1_max.append(loca_acc)
        area_corloc = list(compress(area_hist, corLoc))
        items, counts = np.unique(area_corloc, return_counts=True)
        log.info("[HISTOGRAM STN BOXES]")
        log.info(items)
        log.info(counts)
        if acc > test_acc1:
            test_acc1 = acc
        if loca_acc > best_loc_acc1:
            best_loc_acc1 = loca_acc
            torch.save(model.state_dict(), "/export/livia/Database/ILSVRC_2012_LOC/model_singlescale_ckpt.pth")

        vis_utils_imagenet.plot_bboxes_of_few_images(imdb, detection_max, detection_max_NT, selected_indices, imagepath, epoch, imdb.gt_roidb, "corLoc", image_res)
        loca_acc, corLoc = imagenet_eval.eval(detection_max_NT, imdb.gt_roidb, log)
        loc_test_acc1_max_NT.append(loca_acc)
        area_corloc = list(compress(area_hist, corLoc))
        items, counts = np.unique(area_corloc, return_counts=True)
        log.info("[HISTOGRAM NO TRANSFORM BOXES]")
        log.info(items)
        log.info(counts)
        log.info("Best TEST accuracy so far[full test set (MAX)]: {}".format(test_acc1))
        log.info("Best TEST LOCALIZATION accuracy so far[full test set (MAX)]: {}".format(best_loc_acc1))


def filter_test_for_scale(imdb):
    cache_dir = os.path.join(imdb._devkit_path, 'annotations_cache')
    cachefile = os.path.join(cache_dir, 'test_annots.pkl')
    with open(cachefile, "rb") as f:
        recs = pickle.load(f)
    indicator = np.zeros(len(recs), dtype=np.int32)
    for i in range(len(recs)):
        key_i = recs.keys()[i]
        area = (recs[key_i][3] - recs[key_i][1]) * (recs[key_i][4] - recs[key_i][2])
        if 34635.0 < area < 44133.0:
        #if 6141.0 < area < 15639.0:
            indicator[i] = 1
    imdb._image_names = list(compress(imdb._image_names, indicator))
    imdb._image_index = list(compress(imdb._image_index, indicator))
    imdb._image_labels = list(compress(imdb._image_labels, indicator))
    imdb._annotations = list(compress(imdb._annotations, indicator))
    imdb._image_dims = list(compress(imdb._image_dims, indicator))
    return imdb, indicator


def get_area_hist(imdb):
    """ divide area into 3 bins based on size of the bounding box as S/M/L category with
    [S=205699, M=146168, L=172099] on train set and [S=17197, M=13912, L=18891] on test set"""
    areas = [roidb['area_original'] for roidb in imdb.gt_roidb]
    bins = np.array([10, 50000, 100000, 18000000])
    indicator = np.digitize(areas, bins)
    return indicator


def sample_few_images(imdb):
    max_count = 100
    class_dict = dict(zip(range(imdb.num_classes), np.zeros(imdb.num_classes, dtype=np.int)))
    sampled_gt_roidb = []
    gt_roidb = imdb.gt_roidb
    for i in range(len(gt_roidb)):
        gt_classes = gt_roidb[i]['gt_classes']
        if class_dict[gt_classes[0]] > max_count:
            continue
        else:
            class_dict[gt_classes[0]] += 1
            sampled_gt_roidb.append(gt_roidb[i])
    return sampled_gt_roidb



def main(log, args=None, arglist=None):
    #global loc_test_acc1
    help_text = """ Collect the required arguments """
    parser = argparse.ArgumentParser(description=help_text, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-e", "--num_epochs", type=int, help="number of trainig epochs", default=20)
    parser.add_argument("-bs", "--batch_size", type=int, help="batch size", default=16)
    parser.add_argument("-ir", "--image_res", type=int, help="batch size", default=320)
    parser.add_argument("--plot_every", type=int, help="batch size", default=50)

    if not args:
        args = parser.parse_args(arglist)

    #initialize the model
    model = Net(1001).to(device)
    log.info("Model initialization completed")


    #set the optimizer
    #optimizer = optim.SGD([{"params": model.conv_stn.parameters()},
    #                       {"params": model.localization.parameters(), "weight_decay": 0},
    #                       {"params": model.fc_loc.parameters(), "weight_decay": 0},
    #                       {"params": model.features.parameters()},
    #                       {"params": model.conv_last_10map.parameters()},
    #                       {"params": model.bn_last_10map.parameters()}
    #                       ], lr=1e-4, weight_decay=1e-4, momentum=0.9)
    optimizer = optim.SGD([{"params": model.stn.parameters(), "weight_decay": 0},
                           #{"params": model.stn2.parameters(), "weight_decay": 0},
                           {"params": model.features.parameters()},
                           {"params": model.conv_last_10map.parameters()},
                           {"params": model.bn_last_10map.parameters()}
                           ], lr=1e-4, weight_decay=1e-4, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)
    # load the image databases
    train_imdb = image_net("train")
    train_imdb.gt_roidb = train_imdb.scale_ground_truth_boxes(args.image_res)
    log.info("Train images loaded: {} images".format(len(train_imdb.gt_roidb)))
    test_imdb = image_net("test")
    test_imdb.gt_roidb = test_imdb.scale_ground_truth_boxes(args.image_res)
    log.info("Test images loaded: {} images".format(len(test_imdb.gt_roidb)))
    # rescale the ground truth according to image resolution
    log.info("Image resolution: {}".format(args.image_res))

    train_imdb.gt_roidb = sample_few_images(train_imdb)

    transform_imagenet_train = transforms.Compose([
        transforms.Resize((args.image_res, args.image_res)),
        # transforms.CenterCrop(512),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transform_imagenet_test = transforms.Compose([
        transforms.Resize((args.image_res, args.image_res)),
        # transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    #test_imdb1 = deepcopy(test_imdb)
    #test_imdb1, indicator = filter_test_for_scale(test_imdb1)
    #indicator = np.where(indicator==1)[0]
    area_hist = get_area_hist(test_imdb, args.image_res)
    item, count = np.unique(area_hist, return_counts=True)
    log.info("Histogram of areas")
    log.info(item)
    log.info(count)
    log.info("Train size: {}".format(len(train_imdb.gt_roidb)))
    log.info("Test size: {}".format(len(test_imdb.gt_roidb)))
    #log.info("Test size after picking the selected scale: {}".format(len(test_imdb1._image_index)))


    for epoch in range(args.num_epochs):
        scheduler.step()
        train_model(model, train_imdb, log, optimizer, epoch+1, args.batch_size, transform_imagenet_train, args.image_res)
        test_model(model, test_imdb, log, epoch+1, args.batch_size, area_hist, transform_imagenet_test, args.image_res)

    log.info("Training [CLASSIFICATION] accuracies")
    log.info(class_train_acc)
    log.info("\nTrain [LOC] accuracy with max selection")
    log.info(loc_train_acc_max)
    loc_train_acc = np.array(loc_train_acc_max)
    log.info(np.mean(loc_train_acc))
    log.info("\nTrain [LOC] accuracy with max selection and NO TRANSFORM")
    log.info(loc_train_acc_max_NT)
    loc_train_acc = np.array(loc_train_acc_max_NT)
    log.info(np.mean(loc_train_acc))

    log.info("Test [CLASSIFICATION] accuracies on [FULL SET]")
    log.info(class_test_acc1)
    log.info("\nTest [LOC] accuracy with max selection [FULL SET]")
    log.info(loc_test_acc1_max)
    loc_test_acc = np.array(loc_test_acc1_max)
    log.info(np.mean(loc_test_acc))
    #log.info("\nTest [LOC] accuracy with all boxes [FULL SET]")
    #log.info(loc_test_acc1_all)
    #loc_test_acc = np.array(loc_test_acc1_all)
    #log.info(np.mean(loc_test_acc))
    log.info("\nTest [LOC] accuracy with max selection [FULL SET] NO TRANSFORM")
    log.info(loc_test_acc1_max_NT)
    loc_test_acc = np.array(loc_test_acc1_max_NT)
    log.info(np.mean(loc_test_acc))


if __name__=="__main__":
    log = logging.getLogger('All_Logs')
    log.setLevel(logging.INFO)
    fh = logging.FileHandler('running_imagenet_10fmap.log', mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    log.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.addHandler(ch)
    main(log)
