from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
import logging
import sys
import argparse
from random import shuffle
import os.path as osp
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

from utils import data_utils
from utils import vis_utils
from datasets.factory import get_imdb
from model.vggNet import Net
from model.config import get_output_dir
from utils import box_utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_acc = 0.0
fmap = 16

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
    train_size = len(imdb._image_index)
    imagepath = osp.join(imdb._data_path, "JPEGImages")

    index_shuf = range(len(imdb.roidb))
    shuffle(index_shuf)
    imdb._image_index = [imdb._image_index[i] for i in index_shuf]
    imdb.roidb = [imdb.roidb[i] for i in index_shuf]
    #labels = data_utils.load_pascal_labels(imdb)
    labels = data_utils.load_pascal_labels_binary(imdb)

    num_batches_per_epoch = int(train_size/batch_size) + 1
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num+1)*batch_size, train_size)
        trainX = data_utils.load_batch_data(imagepath, imdb, start_index, end_index)
        trainY = torch.from_numpy(labels[start_index:end_index]).float()
        data, target = trainX.to(device), trainY.to(device)
        #print(target.dtype)

        optimizer.zero_grad()
        likelihood, pos_class_scores, _, _ = model(data, epoch, "train", log, batch_size=data.shape[0])
        #print(likelihood)
        #print(torch.log(target*(likelihood - 0.5) + 0.5))

        #loss = torch.sum(torch.log(target*(likelihood - 0.5) + 0.5))
        loss = torch.sum(-target*torch.log(likelihood))
        loss.backward()
        optimizer.step()
        count += 1
        train_loss += loss.item()*data.shape[0]
        if batch_num % 5 == 0:
            log.info("[TRAIN] Epoch: {} batch: {} loss: {}".format(epoch, count, loss.item()))

        with torch.no_grad():
            scores = (2*likelihood) - 1
            scores = torch.sigmoid(scores)
            pred = (scores >= 0.5)
            #print(pred)
            matches = pred.eq(target>0).sum(dim=1)
            union = (torch.sum(pred, dim=1) + torch.sum(target>0, dim=1) - matches).float()
            matches = matches.float()
            batch_acc = torch.sum(matches/union)/data.shape[0]
            acc += batch_acc

    train_loss /= train_size
    acc /= train_size
    log.info("[TRAIN] loss: {} Accuracy: {}".format(train_loss, acc))


def test_model(model, imdb, log, optimizer, epoch, batch_size):
    global test_acc

    imagepath = osp.join(imdb._data_path, "JPEGImages")
    with torch.no_grad():
        model.eval()
        test_loss = 0.0
        acc = 0.0
        correct = 0
        test_size = len(imdb._image_index)
        #labels = data_utils.load_pascal_labels(imdb)
        labels = data_utils.load_pascal_labels_binary(imdb)
        num_batches_per_epoch = int(test_size/batch_size) + 1
        num_classes = len(imdb.classes)
        detection = {k:{l:{} for l in range(test_size)} for k in range(1, num_classes)}

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num+1)*batch_size, test_size)
            testX = data_utils.load_batch_data(imagepath, imdb, start_index, end_index)
            testY = torch.from_numpy(labels[start_index:end_index]).float()
            data, target = testX.to(device), testY.to(device)

            if correct==0:
                likelihood, pos_class_scores, all_boxes, selected_boxes = model(data, epoch, "test", log, batch_id=1, batch_size=data.shape[0])
            else:
                likelihood, pos_class_scores, all_boxes, selected_boxes = model(data, epoch, "test", log, batch_id=0, batch_size=data.shape[0])

            #save the selected boxes as detection for each image, for each class
            for i in range(len(selected_boxes)):
                detection = box_utils.get_class_specific_boxes(selected_boxes[i], all_boxes, detection, pos_class_scores, i+1, fmap, start_index)

            #loss = torch.sum(torch.log(target*(likelihood - 0.5) + 0.5))
            loss = torch.sum(-target*torch.log(likelihood))
            test_loss += loss.item()*data.shape[0]

            scores = (2*likelihood) - 1
            scores = torch.sigmoid(scores)
            pred = (scores >= 0.5)
            matches = pred.eq(target>0).sum(dim=1)
            union = (torch.sum(pred, dim=1) + torch.sum(target>0, dim=1) - matches).float()
            matches = matches.float()
            batch_acc = torch.sum(matches/union)/data.shape[0]
            acc += batch_acc

        #compute the mAP
        output_dir = get_output_dir(imdb, "vgg")
        imdb.evaluate_detections(detection, output_dir)


        test_loss /= test_size
        acc /= test_size
        if acc>test_acc:
            test_acc = acc
        log.info("[TEST] loss:{} Accuracy: {}".format(test_loss, acc))
        log.info("Best TEST accuracy so far: {}".format(test_acc))    


def main(log, args=None, arglist=None):
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
    optimizer = optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.001, momentum=0.9)
    train_imdb = get_imdb("voc_2007_trainval")
    test_imdb = get_imdb("voc_2007_test")

    for epoch in range(args.num_epochs):
        train_model(model, train_imdb, log, optimizer, epoch+1, args.batch_size)
        test_model(model, test_imdb, log, optimizer, epoch+1, args.batch_size)


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
