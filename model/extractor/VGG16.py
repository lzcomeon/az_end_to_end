from __future__ import absolute_import
from __future__ import division
import torch as t
from torch import nn
from torchvision.models import vgg16
from torchvision.models import resnet101
from lib import array_tool as at


def vgg_16(pretrain=True):


    model = vgg16(pretrained=pretrain)
    # print(model)

    features = list(model.features)[:30]
    # print(features)

    extractor = nn.Sequential(*features)


    return extractor


def resnet(pretrain=True):
    model = resnet101(pretrained=pretrain)
    layer1 = list(model.layer1)
    layer2 = list(model.layer2)
    layer3 = list(model.layer3)
    layer4 = list(model.layer4)
    extractor = nn.Sequential(model.conv1, model.bn1, *layer1, *layer2, *layer3, *layer4)

    return extractor







if __name__ == '__main__':

    import os
    print(os.getcwd())
    from datasets.voc_dataset import read_image
    from datasets.dataset import preprocess

    path = '/home/lz/Lab/pytorch/pytorch_az/datasets/tomato/JPEGImages/001_1.jpg'

    img = read_image(path)
    img = preprocess(img)
    img = at.totensor(img, cuda=False)

    print(img.shape)
    img = img[None]

    fea = resnet()

    features = fea(img)
    print(features.shape)