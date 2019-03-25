from __future__ import absolute_import
from __future__ import division
import torch as t
from collections import namedtuple
from torch import nn
from lib.config import opt
from model.classifier.cls_roi import get_minibatch
from lib.vis_tool import Visualizer
from torch.nn import functional as F
from torchnet.meter import AverageValueMeter
import time
import os
import lib.array_tool as at
import numpy as np
from datasets.dataset import preprocess
from model.test import im_propose
from lib.utils.nms import nms

LossTuple = namedtuple('LossTuple',
                       ['cls_loss',
                        'loc_loss',
                        'total_loss',
                        ])


def nograd(f):
    def new_f(*args,**kwargs):
        with t.no_grad():
           return f(*args,**kwargs)
    return new_f


class Classifier_train(nn.Module):
    def __init__(self, extractor, classifier):
        super(Classifier_train, self).__init__()
        self.extractor = extractor
        self.classifier = classifier

        self.optimizer = self.get_optimizer()

        self.vis = Visualizer(env=opt.cls_env)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}
        self.use_preset('evaluate')



    def forward(self, imgs, bboxes, labels, net):



        features = self.extractor(imgs)

        roi_labels, overlaps, im_rois, bbox_targets, bbox_loss = \
            get_minibatch(net, imgs, bboxes, labels)

        im_rois = at.totensor(im_rois)
        cls_score, cls_bbox = self.classifier(features, im_rois)


        cls_loss = F.cross_entropy(cls_score, roi_labels)

        loc_loss = _fast_rcnn_loc_loss(cls_bbox, bbox_targets, bbox_loss, 1)

        losses = [cls_loss, loc_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses), cls_score, cls_bbox, im_rois

    def use_preset(self, preset):
        """Use the given preset during prediction.

        This method changes values of :obj:`self.nms_thresh` and
        :obj:`self.score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.

        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.

        Args:
            preset ({'visualize', 'evaluate'): A string to determine the
                preset to use.

        """
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    @nograd
    def predict(self, imgs, net=None, sizes=None, visualize=False):
        self.eval()
        sizes = list()
        if visualize:
            self.use_preset('visualize')
            prepared_imgs = list()
            for img in imgs:
                size = img.shape[1:]
                # img = preprocess(at.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
            prepared_imgs = imgs
        bboxes = list()
        labels = list()
        scores = list()
        for img, size in zip(prepared_imgs, sizes):
            img = at.totensor(img[None]).float()
            scale = img.shape[3] / size[1]
            im_rois = im_propose(net, img)
            im_rois = at.totensor(im_rois)
            features = self.extractor(img)
            cls_score, cls_bbox = self.classifier(features, im_rois)
            roi_score = cls_score.data
            roi_loc = cls_bbox.data

            roi = at.tonumpy(im_rois) / scale
            roi_loc = at.tonumpy(roi_loc)



            roi_bbox = _bbox_pred(roi, roi_loc)
            roi_bbox = _clip_boxes(roi_bbox, img.shape)
            prob = at.tonumpy(F.softmax(at.totensor(roi_score), dim=1))

            raw_cls_bbox = at.tonumpy(roi_bbox)
            raw_prob = at.tonumpy(prob)

            bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)

            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset('evaluate')
        self.train()
        return bboxes, labels, scores

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, opt.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, opt.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]

            boxes = np.hstack((cls_bbox_l, prob_l[:, np.newaxis]))
            keep = nms(boxes, self.nms_thresh)

            bbox.append(cls_bbox_l[keep])
            label.append(l * np.ones((len(keep),)))
            score.append(prob_l[keep])

        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)

        return bbox, label, score

















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

    def update_meters(self, losses):
        for k, v in losses._asdict().items():
            v_t = v.detach().cpu()
            if v_t>1:
                self.meters[k].add(1)
                continue
            else:
                self.meters[k].add(v.detach().cpu())

    def reset_meters(self):
        for key, meter_ in self.meters.items():
            meter_.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}


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
        self.vis = Visualizer(env=opt.cls_env)

        return self.optimizer


    def train_step(self, imgs, gt_bboxes, gt_labels, scale, net):
        self.optimizer.zero_grad()
        losses, pred_scores, pred_bbox_delats, rois = \
            self.forward(imgs, gt_bboxes, gt_labels, net)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)

        return losses, pred_scores, pred_bbox_delats, rois


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

def _bbox_pred(boxes, box_deltas):
    """Transform the set of class-agnostic boxes into class-specific boxes
        by applying the predicted offsets (box_deltas)
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + opt.EPS
    heights = boxes[:, 3] - boxes[:, 1] + opt.EPS
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = box_deltas[:, 0::4]
    dy = box_deltas[:, 1::4]
    dw = box_deltas[:, 2::4]
    dh = box_deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)

    # # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def _clip_boxes(boxes, shape):
    """clip boxes to image boundries"""
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0.)
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0.)
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], shape[3])
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], shape[2])

    return boxes