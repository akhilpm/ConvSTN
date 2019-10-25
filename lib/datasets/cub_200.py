from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import subprocess
import uuid
from model.config import cfg
from itertools import compress

class cub_200(imdb):
    def __init__(self, image_set):
        name = "cub_200_2011"
        imdb.__init__(self, name)
        self._devkit_path = self._get_default_path()
        self._data_path = self._devkit_path
        self._image_set = image_set
        self._set_indices = open(self._data_path+"/train_test_split.txt").readlines()
        self._set_indices = [np.int32(line.strip().split()[1]) for line in self._set_indices]
        if self._image_set == "test":
            self._set_indices = np.logical_not(self._set_indices).astype(np.int32)

        #load all class names
        f = open(self._data_path+"/classes.txt").readlines()
        f = [line.strip().split()[1] for line in f]
        self._classes = ["__background__"] + f
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))

        #load all image names
        f = open(self._data_path+"/images.txt").readlines()
        self._image_names = [line.strip().split()[1] for line in f]
        self._image_names = list(compress(self._image_names, self._set_indices))
        self._image_index = [line.strip().split()[0] for line in f]
        self._image_index = list(compress(self._image_index, self._set_indices))

        self._image_labels = open(self._data_path+"/image_class_labels.txt").readlines()
        self._image_labels = [line.strip() for line in self._image_labels]
        self._image_labels = list(compress(self._image_labels, self._set_indices))
        self._image_labels = [np.int(item.split()[1]) for item in self._image_labels]

        self._annotations = open(self._data_path+"/bounding_boxes.txt").readlines()
        self._annotations = [line.strip() for line in self._annotations]
        self._annotations = list(compress(self._annotations, self._set_indices))

        self._image_dims = open(self._data_path+"/image_height_width.txt").readlines()
        self._image_dims = [map(float, line.strip().split()) for line in self._image_dims]
        self._image_dims = list(compress(self._image_dims, self._set_indices))

        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        self.config = {'cleanup': False,
                'use_salt': True,
                'matlab_eval': False,
                'rpn_file': None}

    def _get_default_path(self):
        """
        Return the default path where CUB dataset is stored.
        """
        return os.path.join(cfg.DATA_DIR, 'CUB_200_2011')

    
    def image_path_from_index(self, index):
        """ get image path from its index """
        image_name = self._image_names[index]
        image_path = os.path.join(self._data_path, "images", image_name)
        return image_path



