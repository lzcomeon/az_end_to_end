from __future__ import absolute_import
from __future__ import division
import torch as t
from torch import nn
from model.roi.roi_pool import RoIPoolFunction
from lib.config import opt


class az_layer_bn(nn.Module):
    """using to predict zoom scores, adj_score, adj_bbox
        on the given anchors.

    Args:
        features: 4D feature variable, (N, C, H, W)
        rois:
    """
    def __init__(self, spatial_scale):
        super(az_layer_bn, self).__init__()
        self.roi = RoIPoolFunction(7, 7, spatial_scale=spatial_scale)

        self.fc6 = nn.Linear(512*7*7, 4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.bn6 = nn.BatchNorm1d(4096, momentum=0.5)
        self.fc7_1 = nn.Linear(4096, 1024)
        self.relu7_1 = nn.ReLU(inplace=True)
        self.bn7_1 = nn.BatchNorm1d(1024, momentum=0.5)
        self.adj_bbox = nn.Linear(1024, opt.NUM_SUBREG * 4)
        self.adj_score = nn.Linear(1024, opt.NUM_SUBREG)
        self.fc7_2 = nn.Linear(4096, 256)
        self.relu7_2 = nn.ReLU(inplace=True)
        self.bn7_2 = nn.BatchNorm1d(256, momentum=0.5)
        self.zoom_score = nn.Linear(256, 1)

    def forward(self, features, rois):
        # rois.shape should be (n, 5)
        num_rois = rois.shape[0]
        index_of_image = t.zeros((num_rois, 1), dtype=t.float).cuda()
        rois = t.cat((index_of_image, rois.float()), dim=1)

        rois_out = self.roi(features, rois)
        rois_out = rois_out.view(num_rois, -1)

        fc6 = self.fc6(rois_out)
        fc6 = self.relu6(fc6)
        drop6 = self.bn6(fc6)
        fc7_1 = self.fc7_1(drop6)
        fc7_1 = self.relu7_1(fc7_1)
        drop7_1 = self.bn7_1(fc7_1)
        adj_bbox = self.adj_bbox(drop7_1)
        adj_score = self.adj_score(drop7_1)
        fc7_2 = self.fc7_2(drop6)
        fc7_2 = self.relu7_2(fc7_2)
        drop7_2 = self.bn7_2(fc7_2)
        zoom_score = self.zoom_score(drop7_2)
        zoom_score = t.sigmoid(zoom_score)
        adj_score = t.sigmoid(adj_score)

        return zoom_score, adj_score, adj_bbox







if __name__ == '__main__':
    AZ = az_layer_bn(1/16).cuda()

    import os

    print(os.getcwd())
    from datasets.voc_dataset import read_image
    from model.extractor.VGG16 import vgg_16

    path = '/home/lz/Lab/pytorch/pytorch_az/datasets/tomato/JPEGImages/001_1.jpg'
    img = t.tensor(read_image(path))
    img = t.unsqueeze(img, 0).cuda()
    fea = vgg_16().cuda()
    features = fea(img)

    rois = t.tensor([[0, 0, 100, 32],[0, 0, 100, 32]]).float().cuda()

    zoom, score, bbox = AZ(features, rois)
    print(zoom.shape, score.shape, bbox.shape)
    print(zoom, score, bbox)



