from __future__ import absolute_import
from __future__ import division
from lib.config import opt
import torch as t
from torch import nn
import numpy as np
from lib.utils.nms import nms
import lib.array_tool as at
from lib.vis_tool import Visualizer
import os
import time
from model.az_data_layer.az_im_propose import im_propose
from model.end_to_end.az_roi import generate_az_rois
from lib.utils.cython_bbox import bbox_overlaps
from torch.nn import functional as F
from collections import namedtuple
from torchnet.meter import AverageValueMeter


LossTuple = namedtuple('LossTuple',
                       ['zoom_loss',
                        'az_cls_loss',
                        'az_bbox_loss',
                        'cls_loss',
                        'loc_loss',
                        'total_loss',
                        ])

def nograd(f):
    def new_f(*args,**kwargs):
        with t.no_grad():
           return f(*args,**kwargs)
    return new_f




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
        self.use_preset('evaluate')
        self.optimizer = self.get_optimizer()
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}
        self.vis = Visualizer(env=opt.env)







    @property
    def n_class(self):
        # including background
        return self.classifier.n_class



    def forward(self, x, gt_rois, labels, scale=1.):
        """Forward the whole network

        Args:
            x (autograd.Variable): 4D image variable, (N, C, H, W)
        """
        size = x.shape[2:]
        feature = self.extractor(x)
        adj_labels, zoom_labels, rois_keep, adj_targets, adj_loss_weights, rois= \
            generate_az_rois(size, gt_rois)

        num = rois_keep.shape[0]

        dup_rois = np.vstack((rois_keep, rois))
        zoom_score, adj_score, adj_bbox = self.az(feature, at.totensor(dup_rois))

        adj_bbox = at.tonumpy(adj_bbox)

        ex_rois = _bbox_pred(rois, adj_bbox[num:, :])
        ex_rois = _clip_boxes(ex_rois, size)

        a_tb, c_tb = _unwrap_adj_pred(ex_rois, at.tonumpy(adj_score[num:, :]))

        # Use nms to reduce high overlaps
        # boxes_with_score = np.hstack((a_tb, c_tb[:, np.newaxis]))
        # keep = nms(boxes_with_score, self.nms_thresh)

        max_num = np.minimum(a_tb.shape[0], opt.SEAR_NUM_PROPOSALS)
        indA = np.argsort(-c_tb)
        keep = indA[:max_num]


        ex_rois = a_tb[keep]
        bbox_targets, max_overlaps = add_bbox_regression_targets(ex_rois, gt_rois, labels)
        # (2009, 5), (2009,)
        # print('bbox_targets:', bbox_targets.shape, 'max_overlaps', max_overlaps.shape)

        roi_labels, overlaps, im_rois, bbox_targets, bbox_loss \
            = _sample_rois(ex_rois, gt_rois, bbox_targets, max_overlaps, 16, 32,
                           num_classes=2)
        cls_mean = np.array(self.loc_normalize_mean, np.float32)
        cls_std = np.array(self.loc_normalize_std, np.float32)
        cls_mean = np.tile(cls_mean, self.n_class)
        cls_std = np.tile(cls_std, self.n_class)
        bbox_targets = (bbox_targets - cls_mean)/cls_std
        
        

        im_rois = at.totensor(im_rois)
        cls_score, cls_bbox = self.classifier(feature, im_rois)


        roi_labels = at.totensor(roi_labels).long()
        bbox_targets = at.totensor(bbox_targets)
        bbox_loss = at.totensor(bbox_loss)

        # classifier cls score loss
        cls_loss = F.cross_entropy(cls_score, roi_labels)
        # classifier bounding box loss
        loc_loss = _fast_rcnn_loc_loss(cls_bbox, bbox_targets, bbox_loss, 1)

        # az_zoom_loss
        zoom_labels = at.totensor(zoom_labels)
        zoom_loss = F.binary_cross_entropy(zoom_score[:num].squeeze(), zoom_labels.float())

        # az cls score loss
        adj_labels = at.totensor(adj_labels)
        az_cls_loss = F.binary_cross_entropy(adj_score[:num], adj_labels.float())

        # az bbox loss
        adj_bbox = at.totensor(adj_bbox)
        az_bbox_loss = _fast_rcnn_loc_loss(adj_bbox[:num, :],
                                           at.totensor(adj_targets),
                                           at.totensor(adj_loss_weights),
                                           opt.az_sigma)

        losses = [zoom_loss, az_cls_loss, az_bbox_loss, cls_loss, loc_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses

    def save(self, loss, save_path=None):
        if save_path==None:
            save_path = 'checkpoints/end_to_end/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        timestr = time.strftime('%m%d%H%M')
        save_path = save_path + timestr + '.pkl_' + str(loss)
        t.save(self.state_dict(), save_path)



    @nograd
    def predict(self, imgs, visualize=False):
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

            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset('evaluate')
        self.train()
        return bboxes, labels, scores










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

    def update_meters(self, losses):
        for k, v in losses._asdict().items():
            v_t = v.detach().cpu()
            if v_t > 1:
                continue
            else:
                self.meters[k].add(v_t)

    def reset_meters(self):
        for key, meter_ in self.meters.items():
            meter_.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}


    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify
        special optimizer
        """
        lr = opt.cls_lr
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

def _unwrap_adj_pred(boxes, scores):
    scores = scores.ravel()
    boxes = boxes.reshape(-1, 4)
    heights = boxes[:, 3] - boxes[:, 1] + 1.
    widths = boxes[:, 2] - boxes[:, 0] + 1.

    sides = np.minimum(heights, widths)
    keep_inds = np.where(sides >= opt.SEAR_MIN_SIDE)[0]

    return boxes[keep_inds, :], scores[keep_inds]


def add_bbox_regression_targets(ex_rois, gt_rois, gt_labels):
    gt_rois = at.tonumpy(gt_rois)

    ex_rois = np.vstack((ex_rois, gt_rois))
    bbox_targets, max_overlaps = _compute_targets(ex_rois, gt_rois, gt_labels)

    return bbox_targets, max_overlaps


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""
    # Ensure ROIs are floats
    ex_rois = ex_rois.astype(np.float, copy=False)
    gt_rois = gt_rois.astype(np.float, copy=False)

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = bbox_overlaps(ex_rois, gt_rois)

    if gt_rois.shape[0] == 0:
        max_overlaps = opt.TRAIN_BG_THRESH_LO * np.ones((ex_rois.shape[0],), dtype=np.float32)
        targets = np.zeros((ex_rois.shape[0], 4), dtype=np.float32)
        return targets, max_overlaps
    else:
        max_overlaps = ex_gt_overlaps.max(axis=1)
    pos_inds = np.where(max_overlaps >= opt.TRAIN_BBOX_THRESH)[0]  # 77
    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)  # 2024
    # target rois
    tar_rois = gt_rois[gt_assignment[pos_inds], :]
    # positive examples

    pos_rois = ex_rois[pos_inds, :]
    pos_widths = pos_rois[:, 2] - pos_rois[:, 0] + opt.EPS
    pos_heights = pos_rois[:, 3] - pos_rois[:, 1] + opt.EPS
    pos_ctr_x = pos_rois[:, 0] + 0.5 * pos_widths
    pos_ctr_y = pos_rois[:, 1] + 0.5 * pos_heights

    tar_widths = tar_rois[:, 2] - tar_rois[:, 0] + opt.EPS
    tar_heights = tar_rois[:, 3] - tar_rois[:, 1] + opt.EPS
    tar_ctr_x = tar_rois[:, 0] + 0.5 * tar_widths
    tar_ctr_y = tar_rois[:, 1] + 0.5 * tar_heights

    pos_widths = np.maximum(1, pos_widths)
    pos_heights = np.maximum(1, pos_heights)
    tar_widths = np.maximum(1, tar_widths)
    tar_heights = np.maximum(1, tar_heights)

    targets_dx = (tar_ctr_x - pos_ctr_x) / pos_widths
    targets_dy = (tar_ctr_y - pos_ctr_y) / pos_heights
    targets_dw = np.log(tar_widths / pos_widths)
    targets_dh = np.log(tar_heights / pos_heights)

    targets = np.zeros((ex_rois.shape[0], 5), dtype=np.float32)
    targets[pos_inds, 0] = labels[gt_assignment[pos_inds]]
    targets[pos_inds, 1] = targets_dx
    targets[pos_inds, 2] = targets_dy
    targets[pos_inds, 3] = targets_dw
    targets[pos_inds, 4] = targets_dh
    return targets, max_overlaps

def _sample_rois(regions, gt_rois, targets, overlaps, fg_rois_per_image,
                 rois_per_image, num_classes):
    rois = np.vstack((regions, gt_rois))
    labels = targets[:, 0]


    fg_inds = np.where(overlaps >= opt.TRAIN_FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = int(np.minimum(fg_rois_per_image, fg_inds.size))
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = np.random.choice(fg_inds, size=fg_rois_per_this_image,
                             replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < opt.TRAIN_BG_THRESH_HI) &
                       (overlaps >= opt.TRAIN_BG_THRESH_LO))[0]
    if bg_inds.size == 0:
        bg_inds = np.where(overlaps < opt.TRAIN_FG_THRESH)[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = int(np.minimum(bg_rois_per_this_image,
                                            bg_inds.size))
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image,
                             replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]

    bbox_targets, bbox_loss_weights = \
        _get_bbox_regression_labels(targets[keep_inds, :],
                                    num_classes)

    return labels, overlaps, rois, bbox_targets, bbox_loss_weights

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_loss_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_loss_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_loss_weights[ind, start:end] = [1., 1., 1., 1.]
    return bbox_targets, bbox_loss_weights


