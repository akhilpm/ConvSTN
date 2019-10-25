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
#from model.model_CUB_5fmap_noSTN import Net
from model.model_cub_5fmap_noSTN import Net
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
#fmap = 10 #for VGG
fmap = 5 #for ResNet
max_grad_norm = 10
test_im_index = 30
#selected_indices = [797, 1597, 2397, 3197, 3997, 4797]
selected_indices = [10, 30, 60, 90]
#selected_indices = [10, 30, 600, 900]
class_train_acc = []
loc_train_acc_max_NT = []
loc_train_acc_all_NT = []

class_test_acc1 = []
loc_test_acc1_max_NT = []
loc_test_acc1_all_NT = []

class_test_acc2 = []
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

    detection_max_NT = torch.zeros(train_size, 7) #(columns: image_index, class_index, box_cords, conf_score)
    detection_max_NT[:, 0] = torch.tensor(index_shuf)
    detection_all_NT = {k:{'boxes':1, 'pred':1} for k in range(train_size)}

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
        #likelihood2 = x_noSTN.view(x.shape[0], num_classes, -1).sum(2)

        #loss = torch.sum(torch.log(target*(likelihood - 0.5) + 0.5))
        loss1 = torch.sum(-target*torch.log(likelihood))
        #loss2 = torch.sum(-target*torch.log(likelihood2))
        #loss3 = F.kl_div(x.view(-1), x_noSTN.detach().view(-1), reduction='batchmean')
        loss = loss1 #+ loss2 + loss3
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
            detection_max_NT[start_index:end_index, 1] = pred.view(-1)
            detection_max_NT[start_index:end_index, 2:] = all_boxes[2]

            #idx = 0
            #for i in range(start_index, end_index):
            #    detection_all[i]['pred'] = pred[idx]
            #    detection_all[i]['boxes'] = all_boxes[1][idx]
            #    detection_all_NT[i]['pred'] = pred[idx]
            #    detection_all_NT[i]['boxes'] = all_boxes[3][idx]
            #    idx += 1

            #detection = box_utils.get_detected_boxes_cub(all_boxes, x, detection, likelihood, start_index, data.shape[0])

    #with open("../results/detection.pkl", 'wb') as fid:
    #    pickle.dump(detection, fid, pickle.HIGHEST_PROTOCOL)

    #compute the mAP
    with torch.no_grad():
        #vis_utils_cub.plot_bboxes_of_few_images_cub(imdb, image_index, image_names, detection, selected_indices, imagepath, epoch, "train")
        #output_dir = get_output_dir(imdb, "vggCUB")
        #imdb.evaluate_detections(detection, output_dir, log, "train")
        cache_dir = os.path.join(imdb._devkit_path, 'annotations_cache')
        #box_indices = detection_max[:, -1].numpy().astype(np.int32)
        #unique, counts = np.unique(box_indices, return_counts=True)
        #log.info(zip(unique, counts))
        #print(np.argmax(labels, axis=1)[:20])
        train_loss /= train_size
        acc =  acc / train_size
        log.info("[TRAIN] loss: {} Accuracy: {}".format(train_loss, acc))
        class_train_acc.append(acc)

        loca_acc, corLoc = cub_eval.cub_eval(detection_max_NT, imdb._annotations, imdb._image_labels, imdb._image_dims, imdb._image_index,
            cache_dir, "train",  index_shuf, log)
        loc_train_acc_max_NT.append(loca_acc)

        #loca_acc, corLoc = cub_eval_new.cub_eval(detection_all, imdb._annotations, imdb._image_labels, imdb._image_dims, imdb._image_index,
        #    cache_dir, "train",  index_shuf, log)
        #loc_train_acc_all.append(loca_acc)

        #loca_acc, corLoc = cub_eval_new.cub_eval(detection_all_NT, imdb._annotations, imdb._image_labels, imdb._image_dims, imdb._image_index,
        #    cache_dir, "train",  index_shuf, log)
        #loc_train_acc_all_NT.append(loca_acc)



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

        detection_max_NT = torch.zeros(test_size, 7) #(columns: image_index, class_index, box_cords, conf_score)
        detection_max_NT[:, 0] = torch.tensor(np.arange(test_size))
        detection_all_NT = {k:{'boxes':1, 'pred':1} for k in range(test_size)}

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num+1)*batch_size, test_size)
            testX = data_utils.load_cub_batch_data(imagepath, imdb._image_names, start_index, end_index, flag="test")
            testY = torch.from_numpy(labels[start_index:end_index]).float()
            data, target = testX.to(device), testY.to(device)

            likelihood, x, all_boxes, theta = model(data, epoch, "test", log, batch_id=count+1, batch_size=data.shape[0])

            count += 1
            #loss = torch.sum(torch.log(target*(likelihood - 0.5) + 0.5))
            #likelihood2 = x_noSTN.view(x.shape[0], num_classes, -1).sum(2)

            loss1 = torch.sum(-target*torch.log(likelihood))
            #loss2 = torch.sum(-target*torch.log(likelihood2))
            #loss3 = F.kl_div(x.view(-1), x_noSTN.detach().view(-1), reduction='batchmean')
            loss = loss1 #+ loss2 + loss3

            test_loss += loss.item()
            if batch_num % 5 == 0:
                log.info("[TEST] Epoch: {} batch: {} loss: {}".format(epoch, count, loss.item()))

            pred = torch.argmax(likelihood, dim=1).view(-1, 1)
            batch_acc = torch.sum(pred == torch.argmax(target, dim=1).view(-1, 1)).float().cpu().numpy()
            #print(batch_acc), type(batch_acc)
            acc += batch_acc

            detection_max_NT[start_index:end_index, 1] = pred.view(-1)
            detection_max_NT[start_index:end_index, 2:] = all_boxes[2]

            #idx = 0
            #for i in range(start_index, end_index):
            #    detection_all[i]['pred'] = pred[idx]
            #    detection_all[i]['boxes'] = all_boxes[1][idx]
            #    detection_all_NT[i]['pred'] = pred[idx]
            #    detection_all_NT[i]['boxes'] = all_boxes[3][idx]
            #    idx += 1


        test_loss /= test_size
        acc =  acc / test_size
        log.info("[TEST] loss:{} Accuracy: {}".format(test_loss, acc))
        #compute the mAP
        cache_dir = os.path.join(imdb._devkit_path, 'annotations_cache')
        if indicator is None:
            class_test_acc1.append(acc)

            if acc > test_acc1:
                test_acc1 = acc

            loca_acc, corLoc = cub_eval.cub_eval(detection_max_NT, imdb._annotations, imdb._image_labels, imdb._image_dims,
                imdb._image_index, cache_dir, "test",  np.arange(test_size), log)
            loc_test_acc1_max_NT.append(loca_acc)

            if loca_acc > best_loc_acc1:
                best_loc_acc1 = loca_acc
                torch.save(model.state_dict(), "/export/livia/Database/CUB_200_2011/model_ckpt.pth")

            area_corloc = list(compress(area_hist, corLoc))
            items, counts = np.unique(area_corloc, return_counts=True)
            log.info(items)
            log.info(counts)
            #loca_acc, corLoc = cub_eval_new.cub_eval(detection_all, imdb._annotations, imdb._image_labels, imdb._image_dims,
            #    imdb._image_index, cache_dir, "test",  np.arange(test_size), log)
            #loc_test_acc1_all.append(loca_acc)

            #loca_acc, corLoc = cub_eval_new.cub_eval(detection_all_NT, imdb._annotations, imdb._image_labels, imdb._image_dims,
            #    imdb._image_index, cache_dir, "test",  np.arange(test_size), log)
            #loc_test_acc1_all_NT.append(loca_acc)

            log.info("Best TEST accuracy so far[full test set (MAX)]: {}".format(test_acc1))
            log.info("Best TEST LOCALIZATION accuracy so far[full test set (MAX)]: {}".format(best_loc_acc1))
        else:
            class_test_acc2.append(acc)

            if acc > test_acc2:
                test_acc2 = acc

            loca_acc, corLoc = cub_eval.cub_eval(detection_max_NT, imdb._annotations, imdb._image_labels, imdb._image_dims,
                imdb._image_index, cache_dir, "test",  indicator, log)
            loc_test_acc2_max_NT.append(loca_acc)

            if loca_acc > best_loc_acc2:
                best_loc_acc2 = loca_acc

            #loca_acc, corLoc = cub_eval_new.cub_eval(detection_all, imdb._annotations, imdb._image_labels, imdb._image_dims,
            #    imdb._image_index, cache_dir, "test",  indicator, log)
            #loc_test_acc2_all.append(loca_acc)

            #loca_acc, corLoc = cub_eval_new.cub_eval(detection_all_NT, imdb._annotations, imdb._image_labels, imdb._image_dims,
            #    imdb._image_index, cache_dir, "test",  indicator, log)
            #loc_test_acc2_all_NT.append(loca_acc)

            log.info("Best TEST accuracy so far[fixed scale (MAX)]: {}".format(test_acc2))
            log.info("Best TEST LOCALIZATION accuracy so far[fixed scale (MAX)]: {}".format(best_loc_acc2))



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
    scheduler = MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)
    train_imdb = cub_200("train")
    test_imdb = cub_200("test")
    test_imdb1 = deepcopy(test_imdb)
    test_imdb1, indicator = filter_test_for_scale(test_imdb1)
    indicator = np.where(indicator==1)[0]
    area_hist = get_area_hist(test_imdb)
    log.info("Train size: {}".format(len(train_imdb._image_index)))
    log.info("Test size: {}".format(len(test_imdb._image_index)))
    log.info("Test size after picking the selected scale: {}".format(len(test_imdb1._image_index)))

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
    log.info("\nTrain [LOC] accuracy with max selection and NO TRANSFORM")
    log.info(loc_train_acc_max_NT)
    loc_train_acc = np.array(loc_train_acc_max_NT)
    log.info(np.mean(loc_train_acc))
    log.info("\nTrain [LOC] accuracy with all boxes and NO TRANSFORM")
    log.info(loc_train_acc_all_NT)
    loc_train_acc = np.array(loc_train_acc_all_NT)
    log.info(np.mean(loc_train_acc))

    log.info("Test [CLASSIFICATION] accuracies on [FULL SET]")
    log.info(class_test_acc1)
    log.info("\nTest [LOC] accuracy with max selection [FULL SET] NO TRANSFORM")
    log.info(loc_test_acc1_max_NT)
    loc_test_acc = np.array(loc_test_acc1_max_NT)
    log.info(np.mean(loc_test_acc))
    log.info("\nTest [LOC] accuracy with all boxes [FULL SET] NO TRANSFORM")
    log.info(loc_test_acc1_all_NT)
    loc_test_acc = np.array(loc_test_acc1_all_NT)
    log.info(np.mean(loc_test_acc))

    log.info("Test [CLASSIFICATION] accuracies on [REDUCED SET]")
    log.info(class_test_acc2)
    log.info("\nTest [LOC] accuracy with max selection [REDUCED SET] NO TRANSFORM")
    log.info(loc_test_acc2_max_NT)
    loc_test_acc = np.array(loc_test_acc2_max_NT)
    log.info(np.mean(loc_test_acc))
    log.info("\nTest [LOC] accuracy with all boxes [REDUCED SET] NO TRANSFORM")
    log.info(loc_test_acc2_all_NT)
    loc_test_acc = np.array(loc_test_acc2_all_NT)
    log.info(np.mean(loc_test_acc))

if __name__=="__main__":
    log = logging.getLogger('All_Logs')
    log.setLevel(logging.INFO)
    fh = logging.FileHandler('running_5map_noSTN.log', mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    log.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.addHandler(ch)
    main(log)
