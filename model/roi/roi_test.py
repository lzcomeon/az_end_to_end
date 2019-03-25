from __future__ import absolute_import
from __future__ import division
import torch as t
from model.roi.roi_pool import RoIPoolFunction
from datasets.voc_dataset import read_image
from model.extractor.VGG16 import vgg_16
from torch.autograd import Variable



path = '/home/lz/Lab/pytorch/pytorch_az/datasets/tomato/JPEGImages/001_1.jpg'
img = t.from_numpy(read_image(path))
img = t.unsqueeze(img, 0).cuda()

extra = vgg_16().cuda()
features = extra(img)

print(features.shape)

fake_data = t.range(1,32).view(1,1,4,8)
fake_data = Variable(fake_data).cuda()

roi_pool = RoIPoolFunction(2, 2, spatial_scale=1)


rois = t.tensor([[0, 0, 0, 1, 3]]).float().cuda()
print('fake data:', fake_data)
out = roi_pool(fake_data, rois)

print(out)