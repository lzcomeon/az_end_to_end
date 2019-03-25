import torch as t
import torch.nn as nn
from lib.config import opt
from model.roi.roi_pool import RoIPoolFunction





class cls_layer(nn.Module):



    def __init__(self, spatial_scale):
        super(cls_layer, self).__init__()

        self.n_class = opt.n_class
        self.spatial_scale = spatial_scale
        self.roi = RoIPoolFunction(7, 7, self.spatial_scale)

        self.cls_fc1 = nn.Linear(512*7*7, 4096)
        self.cls_relu1 = nn.ReLU(inplace=True)
        self.cls_drop1 = nn.Dropout(p=0.5)
        self.cls_fc2 = nn.Linear(4096, 4096)
        self.cls_relu2 = nn.ReLU(inplace=True)
        self.cls_drop2 = nn.Dropout(p=0.5)
        self.bbox = nn.Linear(4096, self.n_class*4)
        self.cls_score = nn.Linear(4096, self.n_class)

    def forward(self, features, rois):
        # rois.shape should be (n, 5)
        num_rois = rois.shape[0]
        index_of_image = t.zeros((num_rois, 1), dtype=t.float).cuda()
        rois = t.cat((index_of_image, rois.float()), dim=1)

        rois_out = self.roi(features, rois)
        rois_out = rois_out.view(num_rois, -1)

        cls_fc1 = self.cls_fc1(rois_out)
        cls_relu1 = self.cls_relu1(cls_fc1)
        cls_drop1 = self.cls_drop1(cls_relu1)
        cls_fc2 = self.cls_fc2(cls_drop1)
        cls_relu2 = self.cls_relu2(cls_fc2)
        cls_drop2 = self.cls_drop2(cls_relu2)
        bbox = self.bbox(cls_drop2)
        cls_score = self.cls_score(cls_drop2)

        return cls_score, bbox


if __name__ == '__main__':
    cls = cls_layer(1 / 16).cuda()

    import os

    print(os.getcwd())
    from datasets.voc_dataset import read_image
    from model.extractor.VGG16 import vgg_16
    import lib.array_tool as at

    path = '/home/lz/Lab/pytorch/pytorch_az/datasets/tomato/JPEGImages/001_1.jpg'
    img = read_image(path)
    img = at.totensor(img)
    img = t.unsqueeze(img, 0).cuda()
    fea = vgg_16().cuda()
    features = fea(img)

    rois = t.tensor([[0, 0, 0, 100, 32]]).float().cuda()

    score, bbox = cls(features, rois)
    print(score.shape, bbox.shape)
    print(score, bbox)
