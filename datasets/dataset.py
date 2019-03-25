from __future__ import absolute_import
from __future__ import division
import torch as t
from lib.config import opt
from datasets.voc_dataset import VOCBboxDataset
from torchvision import transforms as tvtsf
from skimage import transform as sktsf
from datasets import util
import numpy as np


def inverse_normalize(img):
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255

def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.39, 0.44, 0.32],
                                std=[0.242, 0.218, 0.261])
    img = normalize(t.from_numpy(img))
    return img.numpy()


def preprocess(img, min_size=500, max_size=1000):
    """Preprocess an image for feature extraction.

        The length of the shorter edge is scaled to :obj:`self.min_size`.
        After the scaling, if the length of the longer edge is longer than
        :param min_size:
        :obj:`self.max_size`, the image is scaled to fit the longer edge
        to :obj:`self.max_size`.

        After resizing the image, the image is subtracted by a mean image value
        :obj:`self.mean`.

        Args:
            img (~numpy.ndarray): An image. This is in CHW and RGB format.
                The range of its value is :math:`[0, 255]`.

        Returns:
            ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    # img = sktsf.resize(img, (C, H * scale, W * scale),  mode='reflect',anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size

    normalize = pytorch_normalze
    return normalize(img)


class Transform(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data, x_random=True):
        img, bbox, label = in_data
        _, H, W = img.shape  # 初始大小
        img = preprocess(img, self.min_size, self.max_size)

        _, o_H, o_W = img.shape  # resize后
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))


        # horizontally flip
        img, params = util.random_flip(img, x_random=x_random, return_param=True)
        bbox = util.flip_bbox(bbox, (o_H, o_W), x_flip=params['x_flip'])
        return img, bbox, label, scale


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir)
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        ori_img, bbox_1, label = self.db.get_example(idx)
        img, bbox, label, scale = self.tsf((ori_img, bbox_1, label))

        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)

class TestDataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir, split='test')
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        ori_img, bbox_1, label = self.db.get_example(idx)
        img, bbox, label, scale = self.tsf((ori_img, bbox_1, label), x_random=False)

        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)



if __name__ == '__main__':
    from lib.config import opt

    dataset = Dataset(opt)
    test = TestDataset(opt)


