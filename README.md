# Convolutional STN for Weakly Supervised Localization
The pyTorch code of our paper titled `Convolutional STN for Weakly Supervised Object Localization`

The experiments are performed on two separate datasets, CUB-200 and ImageNet-1K. There are single scale and multi-scale models(with Feature Pyramid Network). 

To run the code for trainining on CUB-200 dataset
``` python cub_train_multiscale.py -e 20 -bs 16 ```

For training on ImageNet
``` python imagenet_train_multiscale.py -e 20 -bs 16 ```

Follow the similar commands for training single scale models with files `cub_train.py` and `imagenet_train.py`.


The paper is available in [Arxiv](https://arxiv.org/abs/1912.01522)
