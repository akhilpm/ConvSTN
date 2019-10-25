from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
from datasets.imdb import imdb
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import subprocess
import uuid
from model.config import cfg
import pickle
from itertools import compress


class image_net(imdb):
    def __init__(self, image_set):
        name = 'ILSVRC_2012'
        imdb.__init__(self, name)
        self._devkit_path = self._get_default_path()
        self._data_path = self._get_data_path()
        self._image_set = image_set
        self._classes = os.listdir(osp.join(self._devkit_path, 'train'))
        self._classes = ['__background__'] + self._classes
        self._class_to_ind = dict(list(zip(self._classes, list(range(self.num_classes)))))
        self.gt_roidb = []



    def _get_default_path(self):
        """
        Return the default path where CUB dataset is stored.
        """
        return os.path.join(cfg.DATA_DIR, 'ILSVRC_2012_LOC')

    def _get_data_path(self):
        """ returns the path to image files """
        return "/export/livia/data/CommonData/imagenet/imagenet_raw-data/"

    def get_val_dataset(self):
        """ Load the validation images. All the images are stored in a single directory irrespective of their class
            Note: A single image may contain multiple annotations of the same object. """
        cache_file = osp.join(self._devkit_path, 'cache', self.name + '_' + self._image_set + '_gt_roidb.pkl')
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    gt_roidb = pickle.load(fid)
                except:
                    gt_roidb = pickle.load(fid, encoding='bytes')
            print('{} {} gt roidb loaded from {}'.format(self.name, self._image_set, cache_file))
            return gt_roidb
        else:
            annot_path = osp.join(self._devkit_path, "validation")
            annotations = os.listdir(annot_path)
            gt_roidb = [self._load_imagenet_annotation(annot_path, image) for image in annotations]
            if None in gt_roidb:
                gt_roidb.remove(None)
            with open(cache_file, 'wb') as fid:
                pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
            print('wrote validation gt roidb to {}'.format(cache_file))
            return gt_roidb


    def get_train_set(self):
        """ Load the training images. Each class is contained in a separate file.
            First read the class directory and then read all files in that directory """
        cache_file = osp.join(self._devkit_path, 'cache', self.name + '_' + self._image_set + '_gt_roidb.pkl')
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    gt_roidb = pickle.load(fid)
                except:
                    gt_roidb = pickle.load(fid, encoding='bytes')
            print('{} {} gt roidb loaded from {}'.format(self.name, self._image_set, cache_file))
            return gt_roidb
        else:
            gt_roidb = []
            annot_path = osp.join(self._devkit_path, "train")
            class_dir = os.listdir(annot_path)
            for dir in class_dir:
                annot_path = osp.join(self._devkit_path, "train", dir)
                annotations = os.listdir(annot_path)
                roidb = [self._load_imagenet_annotation(annot_path, image) for image in annotations]
                if None in roidb:
                    roidb = filter(lambda a: a != None, roidb)
                gt_roidb += roidb
            with open(cache_file, 'wb') as fid:
                pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
            print('wrote train gt roidb to {}'.format(cache_file))
            return gt_roidb


    def _load_imagenet_annotation(self, annot_path, image):
        filename = osp.join(annot_path, image)
        tree = ET.parse(filename)
        objs = tree.findall('object')

        # Load object bounding boxes into a list.
        num_objs = len(objs)
        boxes = np.zeros((num_objs, 4), dtype=np.int32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            cls = obj.find('name').text
            if cls in self._class_to_ind.keys():
                gt_classes[ix] = self._class_to_ind[cls]
            else:
                #gt_classes[ix] = self._class_to_ind['__background__']
                return None
            boxes[ix, :] = [xmin, ymin, xmax, ymax]
        height = int(tree.find('size').find('height').text)
        width = int(tree.find('size').find('width').text)
        image_filename = tree.find('filename').text
        return {'filename': image_filename,
                'boxes': boxes,
                'gt_classes': gt_classes,
                'height': height,
                'width': width}

    def get_scaled_bbox_coordinates(self, gt_roidb, image_res):
        objs = gt_roidb['boxes']
        gt_classes = gt_roidb['gt_classes']
        height, width = gt_roidb['height'], gt_roidb['width']
        filename = gt_roidb['filename']
        num_objs = objs.shape[0]
        boxes = np.zeros((num_objs, 4), dtype=np.int32)
        i = 0

        # scale coordinates
        for obj in objs:
            x_scale, y_scale = image_res / float(width), image_res / float(height)
            xmin, ymin = int(np.round(obj[0] * x_scale)), int(np.round(obj[1] * y_scale))
            xmax, ymax = int(np.round(obj[2] * x_scale)), int(np.round(obj[3] * y_scale))
            boxes[i, :] = [xmin, ymin, xmax, ymax]
            i += 1
        return {'filename': filename,
                'boxes': boxes,
                'gt_classes': gt_classes,
                'height': image_res,
                'width': image_res}

    def scale_ground_truth_boxes(self, image_res):
        cache_file = osp.join(self._devkit_path, 'cache', self.name + '_' + self._image_set + '_' + str(image_res)+ '_gt_roidb.pkl')

        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    gt_roidb = pickle.load(fid)
                except:
                   gt_roidb = pickle.load(fid, encoding='bytes')
            print('{} {} gt roidb for scale {} loaded from {}'.format(self.name, self._image_set, image_res, cache_file))
            return gt_roidb

        else:
            gt_file = osp.join(self._devkit_path, 'cache', self.name + '_' + self._image_set + '_gt_roidb.pkl')
            with open(gt_file, 'rb') as fid:
                try:
                    gt_roidb = pickle.load(fid)
                except:
                    gt_roidb = pickle.load(fid, encoding='bytes')
            print('{} {} gt roidb loaded from {}'.format(self.name, self._image_set, gt_file))

            for i in range(len(gt_roidb)):
                gt_roidb[i] = self.get_scaled_bbox_coordinates(gt_roidb[i], image_res)
                if i % 1000 == 0:
                    print('Reading annotation for {:d}/{:d}'.format(i + 1, len(gt_roidb)))

            print('Saving cached annotations to {:s}'.format(cache_file))
            with open(cache_file, 'wb') as f:
                pickle.dump(gt_roidb, f)
            return gt_roidb


