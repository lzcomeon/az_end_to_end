from __future__ import absolute_import
from __future__ import division
import torch as t
from lib.config import opt
from torch import nn
from model.az_data_layer.az_im_propose import im_propose
import lib.array_tool as at
import numpy as np
from torch.nn import functional as F
from lib.utils.nms import nms


class GM(nn.Module):
    def __init__(self, extractor, az, classifier,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)
                 ):
        super(GM, self).__init__()

        self.extractor = extractor
        self.az = az
        self.classifier = classifier

        # mean and std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        # self.use_preset('evaluate')
        # self.vis = Visualizer(env=opt.env)

    def forward(self, img, sizes):
        self.eval()

        size = img.shape[1:]
        img = img[None]
        # scale = img.shape[3] / size[1]
        feature = self.extractor(img)
        proposals, score = im_propose(self.az, feature, size, num_proposals=300)
        # bbox_with_scres = np.hstack((proposals, score[:, np.newaxis]))
        # keep_bbox = nms(bbox_with_scres, self.nms_thresh)
        # bbox = at.totensor(proposals[keep_bbox, :])
        pred_scores, pred_bbox = self.classifier(feature, at.totensor(proposals))
        mean = t.Tensor(self.loc_normalize_mean).cuda(). \
            repeat(self.n_class)[None]
        std = t.Tensor(self.loc_normalize_std).cuda(). \
            repeat(self.n_class)[None]
        pred_bbox = (pred_bbox * std + mean)

        pred_bbox = _bbox_pred(at.tonumpy(proposals), at.tonumpy(pred_bbox))
        pred_bbox = _clip_boxes(pred_bbox, size)
        pred_scores = at.tonumpy(F.softmax(pred_scores, dim=1))
        bbox, label, score = self._suppress(pred_bbox, pred_scores)



        return bbox, label, score







    @property
    def n_class(self):
        # including background
        return self.classifier.n_class

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

def _clip_boxes(boxes, size):
    """clip boxes to image boundries"""
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0.)
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0.)
    boxes[:, 0::4] = np.minimum(boxes[:, 0::4], size[1])
    boxes[:, 1::4] = np.minimum(boxes[:, 1::4], size[0])
    boxes[:, 2::4] = np.maximum(boxes[:, 2::4], 0.)
    boxes[:, 3::4] = np.maximum(boxes[:, 3::4], 0.)
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], size[1])
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], size[0])

    return boxes