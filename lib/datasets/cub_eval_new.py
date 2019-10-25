from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
from PIL import Image
import numpy as np


def get_scaled_bbox_coordinates(bbox, class_idx, image_dims):
    cords = bbox.split()
    cords = [float(item) for item in cords]
    xmin, ymin = int(cords[1]), int(cords[2])
    xmax, ymax = int(cords[1] + cords[3]), int(cords[2] + cords[4])

    #scale coordinates
    x_scale, y_scale = 320.0/image_dims[0], 320.0/image_dims[1]
    xmin, ymin = int(np.round(xmin*x_scale)), int(np.round(ymin*y_scale))
    xmax, ymax = int(np.round(xmax*x_scale)), int(np.round(ymax*y_scale))
    obj = np.array([class_idx, xmin, ymin, xmax, ymax])
    return obj




def get_ap(rec, prec):
    """ ap = voc_ap(rec, prec)
    Compute AP given precision and recall.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_iou(BBGT, bb):
    ixmin = np.maximum(BBGT[0], bb[:, 0])
    iymin = np.maximum(BBGT[1], bb[:, 1])
    ixmax = np.minimum(BBGT[2], bb[:, 2])
    iymax = np.minimum(BBGT[3], bb[:, 3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    uni = ((bb[:, 2] - bb[:, 0] + 1.) * (bb[:, 3] - bb[:, 1] + 1.) +
            (BBGT[2] - BBGT[0] + 1.) *
            (BBGT[3] - BBGT[1] + 1.) - inters)
    overlaps = inters / uni
    ovmax = np.max(overlaps)
    no_good_overlap = np.sum(overlaps >= 0.5)
    return ovmax, no_good_overlap


def cub_eval(detection,
            ground_truth,
            image_labels,
            image_dims,
            image_index,
            cachedir,
            imageset,
            index_shuf,
            log,
            ovthresh=0.5):

    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, '%s_annots.pkl' %imageset)
    if not os.path.isfile(cachefile):
        recs = {}

        for i, index in enumerate(image_index):
            class_idx = image_labels[i]
            recs[int(index)-1] = get_scaled_bbox_coordinates(ground_truth[i], class_idx, image_dims[i])
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(i + 1, len(image_index)))

        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)

    else:
        with open(cachefile, "rb") as f:
            try:
                recs = pickle.load(f)
            except:
                recs = pickle.load(f, encoding="bytes")


    default_order = recs.keys()
    #print(default_order[:20])
    #index_shuf = [int(image_index[index])-1 for index in index_shuf]
    #print(index_shuf)
    bounding_box = np.array([recs[default_order[i]] for i in index_shuf])
    #print(bounding_box[:, 0][:20])
    #print(detection)
    data_size = len(index_shuf)
    tp = np.zeros([data_size, 2])
    label_correct = 0
    for i in range(data_size):
        #print(detection[i][1], bounding_box[i][0])
        if detection[i]["pred"] == bounding_box[i][0]:
            label_correct += 1
            BBGT = np.array(bounding_box[i][1:])
            bb = detection[i]['boxes']
            iou, no_good_overlap = compute_iou(BBGT, bb)
            if iou >= ovthresh:
                tp[i][0] = 1
                tp[i][1] = no_good_overlap

    #fp = np.cumsum(fp)
    #tp = np.cumsum(tp)
    #rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    #prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    #ap = get_ap(rec, prec)
    item, counts = np.unique(tp[:, 1], return_counts=True)
    log.info("="*50)
    log.info(item)
    log.info(counts)
    log.info("="*50)
    log.info("bbox correct label: {}".format(float(label_correct)/data_size))
    correct = float(np.sum(tp[:, 0])) / data_size
    log.info("Localization accuracy: {} ({}/{})".format(correct, np.sum(tp[:, 0]), data_size))
    return correct, tp[:, 0]
