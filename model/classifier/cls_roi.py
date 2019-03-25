from __future__ import absolute_import
from __future__ import division
import numpy as np
import lib.utils.cython_div as div
from lib.utils.cython_bbox import bbox_overlaps
import torch as t
from lib.config import opt
from model.az_data_layer.az_with_vgg16 import AZ_With_VGG16
import lib.array_tool as at
from model.test import im_propose









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


def get_minibatch(net, img, bbox, label):
    regions = im_propose(net, img)
    # print('regions', regions.shape, 'gt_rois', bbox.shape)  # (2000,4)
    bbox_targets, max_overlaps = add_bbox_regression_targets(regions, bbox, label)
    # (2009, 5), (2009,)
    # print('bbox_targets:', bbox_targets.shape, 'max_overlaps', max_overlaps.shape)

    labels, overlaps, im_rois, bbox_targets, bbox_loss \
        = _sample_rois(regions, bbox, bbox_targets, max_overlaps, 32, 128,
                       num_classes=2)
    labels = at.totensor(labels).long()
    bbox_targets = at.totensor(bbox_targets)
    bbox_loss = at.totensor(bbox_loss)


    return labels, overlaps, im_rois, bbox_targets, bbox_loss


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
        # print('--------------------------------')
        # print(ind, start,end)
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_loss_weights[ind, start:end] = [1., 1., 1., 1.]
    return bbox_targets, bbox_loss_weights

if __name__ == '__main__':
    model_path = '/home/lz/Lab/pytorch/pytorch_az/checkpoints/Extractor_AZ_BN/03130808.pkl_0.08616127'
    from datasets.dataset import Dataset, inverse_normalize
    import lib.array_tool as at
    from torch.utils import data as data_

    net = AZ_With_VGG16().cuda()
    net.load_state_dict(t.load(model_path))
    net.eval()

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=opt.num_workers)

    for ii, (img, bbox_, label_, scale) in enumerate(dataloader):
        # scale = at.scalar(scale)
        # continue
        img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()

        bbox = bbox[0]
        label = label[0]

        labels, overlaps, im_rois, bbox_targets, bbox_loss = get_minibatch(net, img, bbox, label)

        print(labels.shape, im_rois.shape, bbox_targets.shape, bbox_loss.shape)








