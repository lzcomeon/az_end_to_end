from __future__ import absolute_import
from __future__ import division
import torch as t
from torch import nn
from model.roi.roi_pool import RoIPoolFunction
from lib.config import opt
from model.extractor.VGG16 import vgg_16
from model.az_data_layer.az_forward import az_layer
from model.az_data_layer.az_forward_bn import az_layer_bn
import lib.array_tool as at




class AZ_With_VGG16(nn.Module):



    def __init__(self, bn=True):
        super(AZ_With_VGG16, self).__init__()
        self.extractor = vgg_16(pretrain=False)
        if bn:
            self.az = az_layer_bn(1./16)
        else:
            self.az = az_layer(1./16)


    def forward(self, img, rois):
        features = self.extractor(img)
        rois = at.totensor(rois)
        zoom_score, adj_score, adj_bbox = self.az(features, rois)
        return features, zoom_score, adj_score, adj_bbox





if __name__ == '__main__':
    model_path = '/home/lz/Lab/pytorch/pytorch_az/checkpoints/Extractor_AZ_BN/03050705.pkl_0.15837276'
    net = AZ_With_VGG16().cuda()
    # net.load_state_dict(t.load(model_path))


    from datasets.voc_dataset import read_image
    import lib.array_tool as at


    path = '/home/lz/Lab/pytorch/pytorch_az/datasets/tomato/JPEGImages/001_1.jpg'
    img = read_image(path)
    img = at.totensor(img)
    img = t.unsqueeze(img, 0).cuda()

    rois = t.tensor([[0, 0, 0, 100, 32]]).float().cuda()
    features, zoom, score, bbox = net(img, rois)
    print(features.shape, zoom.shape, score.shape, bbox.shape)
    print(zoom, score, bbox)

