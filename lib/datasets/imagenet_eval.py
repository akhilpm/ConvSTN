from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
from PIL import Image
import numpy as np


def compute_iou(BBGT, bb):
    ixmin = np.maximum(BBGT[:, 0], bb[0])
    iymin = np.maximum(BBGT[:, 1], bb[1])
    ixmax = np.minimum(BBGT[:, 2], bb[2])
    iymax = np.minimum(BBGT[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
            (BBGT[:, 2] - BBGT[:, 0] + 1.) *
            (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
    overlap = inters / uni
    return overlap


def eval(detection,
            gt_roidb,
            log,
            ovthresh=0.5):

    #index_shuf = [int(image_index[index])-1 for index in index_shuf]
    #print(index_shuf)
    data_size = len(gt_roidb)
    tp = np.zeros([data_size, 2])
    label_correct = 0
    for i in range(data_size):
        #print(detection[i][1], bounding_box[i][0])
        BBGT = gt_roidb[i]['boxes']
        bb = detection[i][2:6].cpu().numpy()
        iou = compute_iou(BBGT, bb)
        if detection[i][1] == gt_roidb[i]['gt_classes'][0]:
            label_correct += 1
            if (iou >= ovthresh).any():
                tp[i][0] = 1
        if (iou >= ovthresh).any():
            tp[i][1] = 1

    #fp = np.cumsum(fp)
    #tp = np.cumsum(tp)
    #rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    #prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    #ap = get_ap(rec, prec)
    log.info("bbox correct label: {}".format(float(label_correct)/data_size))
    log.info("bbox only correct localization: {} ".format(np.sum(tp[:, 1])))
    correct = float(np.sum(tp[:, 0])) / data_size
    log.info("Localization accuracy: {} ({}/{})".format(correct, np.sum(tp[:, 0]), data_size))
    return correct, tp[:, 0]
