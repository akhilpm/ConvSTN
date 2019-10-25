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
from random import shuffle
import os
import os.path as osp
from PIL import Image
from torch.optim.lr_scheduler import MultiStepLR
from copy import deepcopy

from utils import data_utils
from utils import vis_utils_cub
from datasets.factory import get_imdb
from model.model_CUB2 import Net
#from model.model_CUB_FC2 import Net
#from model.model_CUB_FC_4param import Net
from model.config import get_output_dir
from utils import box_utils_cub
from datasets.cub_200 import cub_200
from datasets import cub_eval
from datasets import cub_eval_new

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_acc1 = 0.0
best_loc_acc1 = 0.0
test_acc2 = 0.0
best_loc_acc2 = 0.0
test_im_index = 90
fmap = 10 #for VGG
#fmap = 5 #for ResNet
max_grad_norm = 10
test_im_index = 30
#selected_indices = [797, 1597, 2397, 3197, 3997, 4797]
selected_indices = [10, 30, 60, 90]
#selected_indices = [10, 30, 600, 900]
class_train_acc = []
loc_train_acc_max = []
loc_train_acc_all = []

class_test_acc1 = []
loc_test_acc1_max = []
loc_test_acc1_all = []

class_test_acc2 = []
loc_test_acc2_max = []
loc_test_acc2_all = []

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


def train_model(model, imdb, log, optimizer, epoch, batch_size):
    model.train()
    correct = 0
    count = 0
    train_loss = 0.0
    acc = 0.0
    theta_start = int(test_im_index/batch_size)
    offset = test_im_index - theta_start*batch_size
    #imdb._image_index = imdb._image_index[:100]
    #imdb._image_labels = imdb._image_labels[:100]
    train_size = len(imdb._image_index)
    #train_scores = np.zeros([0, imdb.num_classes])
    imagepath = osp.join(imdb._data_path, "images")

    index_shuf = np.random.permutation(train_size)
    #index_shuf = [int(imdb._image_index[index])-1 for index in index_shuf]

    image_index = [imdb._image_index[i] for i in index_shuf]
    image_labels = [imdb._image_labels[i] for i in index_shuf]
    image_names = [imdb._image_names[i] for i in index_shuf]
    #labels = data_utils.load_pascal_labels(imdb)
    labels = data_utils.load_cub_labels_binary(image_index, image_labels)
    num_classes = len(imdb.classes)


    #image = image_index[test_im_index]
    #image_at_path = osp.join(imagepath, image_names[test_im_index])
    #rgb_image = Image.open(image_at_path).resize((320, 320), Image.BILINEAR).convert("RGB")
    #detection = {k:{l:torch.zeros(0, 5) for l in range(train_size)} for k in range(1, num_classes)}
    detection_max = torch.zeros(train_size, 7) #(columns: image_index, class_index, box_cords, conf_score)
    detection_max[:, 0] = torch.tensor(index_shuf)
    detection_all = {k:{'boxes':1, 'pred':1} for k in range(train_size)}

    #with open("../results/detection_empty.pkl", 'wb') as fid:
    #    pickle.dump(detection, fid, pickle.HIGHEST_PROTOCOL)

    if train_size%batch_size==0:
        num_batches_per_epoch = int(train_size/batch_size)
    else:
        num_batches_per_epoch = int(train_size/batch_size) + 1
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num+1)*batch_size, train_size)
        trainX = data_utils.load_cub_batch_data(imagepath, image_names, start_index, end_index)
        trainY = torch.from_numpy(labels[start_index:end_index]).float()
        data, target = trainX.to(device), trainY.to(device)
        #print(target)

        optimizer.zero_grad()
        likelihood, x, all_boxes, theta = model(data, epoch, "train", log, batch_id=count+1, batch_size=data.shape[0])
        #train_scores = np.append(train_scores, likelihood.cpu().detach().numpy(), axis=0)
        #print(likelihood)
        #print(torch.log(target*(likelihood - 0.5) + 0.5))

        #loss = torch.sum(torch.log(target*(likelihood - 0.5) + 0.5))
        loss = torch.sum(-target*torch.log(likelihood))
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        #torch.nn.utils.clip_grad_value_(model.parameters(), 100.0)
        optimizer.step()
        count += 1
        train_loss += loss.item()
        if batch_num % 5 == 0:
            log.info("[TRAIN] Epoch: {} batch: {} loss: {}".format(epoch, count, loss.item()))
            #print(torch.max(likelihood, dim=1))
            #print(torch.argmax(trainY, dim=1))

        with torch.no_grad():
            pred = torch.argmax(likelihood, dim=1).view(-1, 1)
            batch_acc = torch.sum(pred == torch.argmax(target, dim=1).view(-1, 1)).float().cpu().numpy()
            acc += batch_acc
            detection_max[start_index:end_index, 1] = pred.view(-1)
            detection_max[start_index:end_index, 2:] = all_boxes[0]

            idx = 0
            for i in range(start_index, end_index):
                detection_all[i]['pred'] = pred[idx]
                detection_all[i]['boxes'] = all_boxes[1][idx]
                idx += 1


    #with open("../results/detection.pkl", 'wb') as fid:
    #    pickle.dump(detection, fid, pickle.HIGHEST_PROTOCOL)

    #compute the mAP
    with torch.no_grad():
        #vis_utils_cub.plot_bboxes_of_few_images_cub(imdb, image_index, image_names, detection, selected_indices, imagepath, epoch, "train")
        #output_dir = get_output_dir(imdb, "vggCUB")
        #imdb.evaluate_detections(detection, output_dir, log, "train")
        cache_dir = os.path.join(imdb._devkit_path, 'annotations_cache')
        train_loss /= train_size
        acc =  acc / train_size
        log.info("[TRAIN] loss: {} Accuracy: {}".format(train_loss, acc))
        class_train_acc.append(acc)

        loca_acc, corLoc = cub_eval.cub_eval(detection_max, imdb._annotations, imdb._image_labels, imdb._image_dims, imdb._image_index, 
            cache_dir, "train",  index_shuf, log)
        loc_train_acc_max.append(loca_acc)

        loca_acc, corLoc = cub_eval_new.cub_eval(detection_all, imdb._annotations, imdb._image_labels, imdb._image_dims, imdb._image_index, 
            cache_dir, "train",  index_shuf, log)
        loc_train_acc_all.append(loca_acc)

        #box_indices = detection[:, -1].numpy().astype(np.int32)
        #unique, counts = np.unique(box_indices, return_counts=True)
        #log.info(zip(unique, counts))
        #print(np.argmax(labels, axis=1)[:20])



def test_model(model, imdb, log, optimizer, epoch, batch_size, area_hist, indicator=None):
    global test_acc1, test_acc2
    global best_loc_acc1, best_loc_acc2
    global test_im_index

    theta_start = int(test_im_index/batch_size)
    offset = test_im_index - theta_start*batch_size
    #imdb._image_index = imdb._image_index[:100]
    #imdb._image_labels = imdb._image_labels[:100]

    imagepath = osp.join(imdb._data_path, "images")
    with torch.no_grad():
        model.eval()
        test_loss = 0.0
        acc = 0.0
        correct = 0
        count = 0
        test_size = len(imdb._image_index)
        #labels = data_utils.load_pascal_labels(imdb)
        labels = data_utils.load_cub_labels_binary(imdb._image_index, imdb._image_labels)

        if test_size%batch_size==0:
            num_batches_per_epoch = int(test_size/batch_size)
        else:
            num_batches_per_epoch = int(test_size/batch_size) + 1

        num_classes = len(imdb.classes)
        detection_max = torch.zeros(test_size, 7) #(columns: image_index, class_index, box_cords, conf_score)
        detection_max[:, 0] = torch.tensor(np.arange(test_size))
        detection_all = {k:{'boxes':1, 'pred':1} for k in range(test_size)}

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num+1)*batch_size, test_size)
            testX = data_utils.load_cub_batch_data(imagepath, imdb._image_names, start_index, end_index)
            testY = torch.from_numpy(labels[start_index:end_index]).float()
            data, target = testX.to(device), testY.to(device)

            likelihood, x, all_boxes, theta = model(data, epoch, "test", log, batch_id=count+1, batch_size=data.shape[0])

            #if batch_num==theta_start:
            #    theta_ind = offset*(fmap**2)
            #    theta_stop = (offset+1)*(fmap**2)
            #    box_utils_cub.plot_all_boxes(model.box_pos, theta[theta_ind:theta_stop], model.grid_positions_is[0], 
            #        pos_class_scores[offset], rgb_image, fmap, 320, epoch)

            #save the selected boxes as detection for each image, for each class
            #for i in range(len(selected_boxes)):
            #    detection = box_utils.get_class_specific_boxes(selected_boxes[i], all_boxes, detection, pos_class_scores, i+1, fmap, start_index)
            #detection = box_utils.get_detected_boxes_cub(all_boxes, pos_class_scores, detection, likelihood, start_index, data.shape[0])

            count += 1
            #loss = torch.sum(torch.log(target*(likelihood - 0.5) + 0.5))
            loss = torch.sum(-target*torch.log(likelihood))
            test_loss += loss.item()
            if batch_num % 5 == 0:
                log.info("[TEST] Epoch: {} batch: {} loss: {}".format(epoch, count, loss.item()))

            pred = torch.argmax(likelihood, dim=1).view(-1, 1)
            batch_acc = torch.sum(pred == torch.argmax(target, dim=1).view(-1, 1)).float().cpu().numpy()
            acc += batch_acc

            detection_max[start_index:end_index, 1] = pred.view(-1)
            detection_max[start_index:end_index, 2:] = all_boxes[0]

            idx = 0
            for i in range(start_index, end_index):
                detection_all[i]['pred'] = pred[idx]
                detection_all[i]['boxes'] = all_boxes[1][idx]
                idx += 1


        test_loss /= test_size
        acc =  acc / test_size
        log.info("[TEST] loss:{} Accuracy: {}".format(test_loss, acc))
        #compute the mAP
        #vis_utils_cub.plot_bboxes_of_few_images_cub(imdb, detection, model.box_pos, selected_indices, imagepath, epoch, "test", indicator=indicator)
        #output_dir = get_output_dir(imdb, "vggCUB")
        cache_dir = os.path.join(imdb._devkit_path, 'annotations_cache')
        if indicator is None:
            class_test_acc1.append(acc)
            loca_acc, corLoc = cub_eval.cub_eval(detection_max, imdb._annotations, imdb._image_labels, 
                imdb._image_dims, imdb._image_index, cache_dir, "test",  np.arange(test_size), log)
            loc_test_acc1_max.append(loca_acc)
            #area_corloc = list(compress(area_hist, corLoc))
            #items, counts = np.unique(area_corloc, return_counts=True)
            #log.info(items)
            #log.info(counts)

            if acc > test_acc1:
                test_acc1 = acc
            if loca_acc > best_loc_acc1:
                best_loc_acc1 = loca_acc

            loca_acc, corLoc = cub_eval_new.cub_eval(detection_all, imdb._annotations, imdb._image_labels, imdb._image_dims, 
                imdb._image_index, cache_dir, "test",  np.arange(test_size), log)
            loc_test_acc1_all.append(loca_acc)

            log.info("Best TEST accuracy so far[full test set (MAX)]: {}".format(test_acc1))
            log.info("Best TEST LOCALIZATION accuracy so far[full test set (MAX)]: {}".format(best_loc_acc1))
        else:
            class_test_acc2.append(acc)
            loca_acc, corLoc = cub_eval.cub_eval(detection_max, imdb._annotations, imdb._image_labels, imdb._image_dims, 
                imdb._image_index, cache_dir, "test",  indicator, log)
            loc_test_acc2_max.append(loca_acc)

            if acc > test_acc2:
                test_acc2 = acc
            if loca_acc > best_loc_acc2:
                best_loc_acc2 = loca_acc
                #indices = np.arange(test_size)
                #vis_utils_cub.plot_bboxes_of_few_images_cub(imdb, detection, model.box_pos, indices, imagepath, epoch, "corloc", "corLoc", indicator)

            loca_acc, corLoc = cub_eval_new.cub_eval(detection_all, imdb._annotations, imdb._image_labels, imdb._image_dims, 
                imdb._image_index, cache_dir, "test",  indicator, log)
            loc_test_acc2_all.append(loca_acc)

            log.info("Best TEST accuracy so far[fixed scale (MAX)]: {}".format(test_acc2))
            log.info("Best TEST LOCALIZATION accuracy so far[fixed scale (MAX)]: {}".format(best_loc_acc2))

        #box_indices = detection[:, -1].numpy().astype(np.int32)
        #unique, counts = np.unique(box_indices, return_counts=True)
        #log.info(zip(unique, counts))

        #area_corloc = list(compress(area_hist, corLoc))
        #items, counts = np.unique(area_corloc, return_counts=True)
        #log.info(items)
        #log.info(counts)

        #np.save("corloc", corLoc)
        #if epoch==5:
            #indices = np.where(corLoc==1)[0]
            #indices = np.arange(test_size)
            #vis_utils_cub.plot_bboxes_of_few_images_cub(imdb, detection, model.box_pos, indices, imagepath, epoch, "corloc", "corLoc", indicator)
            #vis_utils_cub.plot_bboxes_of_few_images_cub(imdb, detection, model.box_pos, indices, imagepath, epoch, "corloc", "corLoc", indices)
        #log.info("Best TEST accuracy so far: {}".format(test_acc))
        #log.info("Best TEST LOCALIZATION accuracy so far: {}".format(best_loc_acc))


def filter_test_for_scale(imdb):
    cache_dir = os.path.join(imdb._devkit_path, 'annotations_cache')
    cachefile = os.path.join(cache_dir, 'test_annots.pkl')
    with open(cachefile, "rb") as f:
        recs = pickle.load(f)
    indicator = np.zeros(len(recs), dtype=np.int32)
    for i in range(len(recs)):
        key_i = recs.keys()[i]
        area = (recs[key_i][3] - recs[key_i][1]) * (recs[key_i][4] - recs[key_i][2])
        #if 34635.0 < area < 44133.0:
        if 6141.0 < area < 15639.0:
            indicator[i] = 1
    imdb._image_names = list(compress(imdb._image_names, indicator))
    imdb._image_index = list(compress(imdb._image_index, indicator))
    imdb._image_labels = list(compress(imdb._image_labels, indicator))
    imdb._annotations = list(compress(imdb._annotations, indicator))
    imdb._image_dims = list(compress(imdb._image_dims, indicator))
    return imdb, indicator


def get_area_hist(imdb):
    cache_dir = os.path.join(imdb._devkit_path, 'annotations_cache')
    cachefile = os.path.join(cache_dir, 'test_annots.pkl')
    with open(cachefile, "rb") as f:
        recs = pickle.load(f)
    indicator = np.zeros(len(recs), dtype=np.int32)
    for i in range(len(recs)):
        key_i = recs.keys()[i]
        area = (recs[key_i][3] - recs[key_i][1]) * (recs[key_i][4] - recs[key_i][2])
        if 6141.0 < area < 15639.1:
            indicator[i] = 1
        elif 15639.2 < area < 25137.3:
            indicator[i] = 2
        elif 23137.4 < area < 34635.5:
            indicator[i] = 3
        elif 34635.6 < area < 44133.7:
            indicator[i] = 4
        elif 44133.8 < area < 53632.0:
            indicator[i] = 5
        elif 53632.1 < area < 63130.1:
            indicator[i] = 6
        elif 63130.2 < area < 72628.3:
            indicator[i] = 7
        elif 72628.4 < area < 82126.5:
            indicator[i] = 8
        elif 82126.6 < area < 91624.7:
            indicator[i] = 9
        elif 91624.8 < area < 101123.0:
            indicator[i] = 10
    return indicator



def main(log, args=None, arglist=None):
    #global loc_test_acc1
    help_text = """ Collect the required arguments """
    parser = argparse.ArgumentParser(description=help_text, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-e", "--num_epochs", type=int, help="number of trainig epochs", default=20)
    parser.add_argument("-bs", "--batch_size", type=int, help="batch size", default=16)
    parser.add_argument("--plot_every", type=int, help="batch size", default=50)

    if not args:
        args = parser.parse_args(arglist)

    #initialize the model
    model = Net().to(device)
    log.info("Model initialization completed")
    #log.info('Data generation completed')

    #set the optimizer
    optimizer = optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-4, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
    train_imdb = cub_200("train")
    test_imdb = cub_200("test")
    test_imdb1 = deepcopy(test_imdb)
    test_imdb1, indicator = filter_test_for_scale(test_imdb1)
    indicator = np.where(indicator==1)[0]
    area_hist = get_area_hist(test_imdb)
    log.info("Train size: {}".format(len(train_imdb._image_index)))
    log.info("Test size: {}".format(len(test_imdb._image_index)))
    log.info("Test size after filter: {}".format(len(test_imdb1._image_index)))

    #imagepath = osp.join(test_imdb._data_path, "images")
    #image = test_imdb._image_index[test_im_index]
    #image_at_path = osp.join(imagepath, test_imdb._image_names[test_im_index])
    #rgb_image = Image.open(image_at_path).resize((320, 320), Image.BILINEAR).convert("RGB")

    for epoch in range(args.num_epochs):
        scheduler.step()
        train_model(model, train_imdb, log, optimizer, epoch+1, args.batch_size)
        test_model(model, test_imdb1, log, optimizer, epoch+1, args.batch_size, area_hist, indicator)
        test_model(model, test_imdb, log, optimizer, epoch+1, args.batch_size, area_hist, indicator=None)

    log.info("Training [CLASSIFICATION] accuracies")
    log.info(class_train_acc)
    log.info("\nTrain [LOC] accuracy with max selection")
    log.info(loc_train_acc_max)
    loc_train_acc = np.array(loc_train_acc_max)
    log.info(np.mean(loc_train_acc))
    log.info("\nTrain [LOC] accuracy with all boxes")
    log.info(loc_train_acc_all)
    loc_train_acc = np.array(loc_train_acc_all)
    log.info(np.mean(loc_train_acc))

    log.info("Test [CLASSIFICATION] accuracies on [FULL SET]")
    log.info(class_test_acc1)
    log.info("\nTest [LOC] accuracy with max selection [FULL SET]")
    log.info(loc_test_acc1_max)
    loc_test_acc = np.array(loc_test_acc1_max)
    log.info(np.mean(loc_test_acc))
    log.info("\nTest [LOC] accuracy with all boxes [FULL SET]")
    log.info(loc_test_acc1_all)
    loc_test_acc = np.array(loc_test_acc1_all)
    log.info(np.mean(loc_test_acc))

    log.info("Test [CLASSIFICATION] accuracies on [REDUCED SET]")
    log.info(class_test_acc2)
    log.info("\nTest [LOC] accuracy with max selection [REDUCED SET]")
    log.info(loc_test_acc2_max)
    loc_test_acc = np.array(loc_test_acc2_max)
    log.info(np.mean(loc_test_acc))
    log.info("\nTest [LOC] accuracy with all boxes [REDUCED SET]")
    log.info(loc_test_acc2_all)
    loc_test_acc = np.array(loc_test_acc2_all)
    log.info(np.mean(loc_test_acc))   


if __name__=="__main__":
    log = logging.getLogger('All_Logs')
    log.setLevel(logging.INFO)
    fh = logging.FileHandler('running.log', mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    log.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.addHandler(ch)
    main(log)
