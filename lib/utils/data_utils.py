import numpy as np
import torch
#import _init_paths
from datasets.factory import get_imdb
import os.path as osp
from PIL import Image
from torchvision import transforms
import gzip

transform = transforms.Compose([
            transforms.Resize((512, 512)),
            #transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])



def load_pascal_labels(imdb):
    no_classes = 21
    no_images = len(imdb._image_index)
    labels = np.ones([no_images, no_classes])*-1.0
    for i, roidb in enumerate(imdb.roidb):
        labels[i][roidb["gt_classes"]] = 1
    #labels = labels[:, 1:]
    labels[:, 0] = labels[:, 0]*-1
    return labels


def load_pascal_labels_binary(imdb):
    no_classes = 21
    no_images = len(imdb._image_index)
    labels = np.zeros([no_images, no_classes])
    for i, roidb in enumerate(imdb.roidb):
        labels[i][roidb["gt_classes"]] = 1
    #labels = labels[:, 1:]
    #labels[:, 0] = 1
    return labels


def load_batch_data(imagepath, imdb, start_index, end_index):
    count = 0
    no_images = end_index - start_index
    images = torch.zeros(no_images, 3, 512, 512)

    for i in range(start_index, end_index):
        image = imdb._image_index[i]
        image_at_path = osp.join(imagepath, image+'.jpg')
        rgb_image = Image.open(image_at_path).convert("RGB")
        #print(transform(rgb_image).shape)
        images[count] = transform(rgb_image)
        count += 1

    return images


def load_cub_labels_binary(image_index, image_labels):
    no_classes = 201
    no_images = len(image_index)
    labels = np.zeros([no_images, no_classes])
    for i, label in enumerate(image_labels):
        labels[i][label] = 1
    #labels = labels[:, 1:]
    #labels[:, 0] = 1
    return labels


def load_cub_batch_data(imagepath, image_names, start_index, end_index, transform, image_size):
    count = 0
    no_images = end_index - start_index
    images = torch.zeros(no_images, 3, image_size, image_size)

    for i in range(start_index, end_index):
        image_at_path = osp.join(imagepath, image_names[i])
        rgb_image = Image.open(image_at_path).convert("RGB")
        #print(transform(rgb_image).shape)
        images[count] = transform(rgb_image)
        count += 1

    return images


def load_imagenet_data(imagepath, imdb, gt_roidb, start_index, end_index, transform, image_size, flag):
    count = 0
    no_images = end_index - start_index
    images = torch.zeros(no_images, 3, image_size, image_size)
    labels = torch.zeros(no_images, imdb.num_classes)

    for i in range(start_index, end_index):
        roidb = gt_roidb[i]
        filename = roidb['filename']
        gt_class = roidb['gt_classes'][0]
        labels[count, gt_class] = 1
        if flag != "train":
            class_name = imdb._classes[gt_class]
            gz_path = osp.join(imagepath, class_name, filename + '.JPEG.gz')
            with gzip.GzipFile(gz_path, 'r') as f:
                rgb_image = Image.open(f).convert("RGB")
        else:
            class_name = filename.split('_')[0]
            gz_path = osp.join(imagepath, class_name, filename + '.JPEG.gz')
            with gzip.GzipFile(gz_path, 'r') as f:
                rgb_image = Image.open(f).convert("RGB")
        images[count] = transform(rgb_image)
        count += 1

    return images, labels
