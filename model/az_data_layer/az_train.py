from __future__ import absolute_import
from __future__ import division
import torch as t
from collections import namedtuple
from torch import nn
from torchnet.meter import AverageValueMeter
from lib.config import opt
from model.az_data_layer.az_roi import generate_az_rois
import lib.array_tool as at
from torch.nn import functional as F
from lib.vis_tool import Visualizer
import time
import os

LossTuple = namedtuple('LossTuple',
                       ['zoom_loss',
                        'az_cls_loss',
                        'az_loc_loss',
                        'total_loss',
                        ])


class AZ_train(nn.Module):
    def __init__(self, extractor, az):
        super(AZ_train, self).__init__()
        self.extractor = extractor
        self.az = az

        self.optimizer = self.get_optimizer()

        self.vis = Visualizer(env=opt.env)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}



    def forward(self, imgs, bboxes, labels, scale):
        """Forward AZ-Net


        Args:
            imgs (autograd.Variable): 4D image variable, (N, C, H, W)
            bboxes: gt_bbox
            labels: gt_labels


        """
        # print('imgs:',imgs.shape, 'bboxes:', bboxes.shape, 'labels:', labels.shape, 'scale', scale)
        features = self.extractor(imgs)
        size = imgs.shape[2:]
        bboxes = bboxes[0]
        labels = labels[0]
        adj_labels, zoom_labels, im_rois, adj_targets, adj_loss_weight = generate_az_rois(size, bboxes)
        im_rois = at.totensor(im_rois)
        zoom_scores, adj_scores, adj_bboxes = self.az(features, im_rois)
        # print('get feature')

        adj_labels = at.totensor(adj_labels)  # 64 x 11
        adj_targets = at.totensor(adj_targets)  # 64 x 44
        adj_loss_weight = at.totensor(adj_loss_weight)  # 64 x 44
        zoom_labels = at.totensor(zoom_labels)  # 64

        az_loc_loss = _fast_rcnn_loc_loss(adj_bboxes,
                                          adj_targets,
                                          adj_loss_weight,
                                          opt.az_sigma)
        az_cls_loss = F.binary_cross_entropy_with_logits(adj_scores, adj_labels.float())

        # ------------------AZ-Net zoom losses----------------------
        zoom_loss = F.binary_cross_entropy_with_logits(zoom_scores.squeeze(), zoom_labels.float())

        # zoom_pred = zoom_scores.squeeze() > 0.5
        # self.zoom_cm.add(zoom_pred, at.totensor(zoom_labels, False))

        losses = [zoom_loss, az_cls_loss, az_loc_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses), zoom_labels, im_rois








    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses, zoom_labels, im_rois= self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses, zoom_labels, im_rois

    def update_meters(self, losses):
        for k, v in losses._asdict().items():
            v_t = v.detach().cpu()
            if v_t>1:
                # self.meters[k].add(1)
                continue
            else:
                self.meters[k].add(v.detach().cpu())

    def reset_meters(self):
        for key, meter_ in self.meters.items():
            meter_.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}


    def save(self, loss, save_path=None ):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.

        Returns:
            save_path(str): the path to save models.
        """

        if save_path == None:
            save_path = 'checkpoints/Extractor_AZ/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        timestr = time.strftime('%m%d%H%M')
        save_path = save_path+timestr+'.pkl_' + str(loss)
        t.save(self.state_dict(), save_path)

        return save_path

    def get_optimizer(self):

        lr = opt.az_lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]

        self.optimizer = t.optim.Adam(params)

        return self.optimizer





    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer






def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    # in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation,
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    # in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1

    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, gt_label, sigma)
    # print('****', loc_loss)
    # Normalize by total number of negtive and positive rois.
    if (gt_label > 0).sum() ==0:
        return t.tensor(0.).cuda()
    else:

        loc_loss /= ((gt_label > 0).sum().float())  # ignore gt_label==-1 for rpn_loss
        loc_loss = loc_loss * 4
        return loc_loss



