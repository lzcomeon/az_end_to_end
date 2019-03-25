from __future__ import absolute_import
from __future__ import division
from lib.config import opt
import numpy as np
from lib import array_tool as at
from lib.utils.cython_bbox import bbox_zoom_labels as bbox_zoom_labels_cy
import lib.utils.cython_div as div
from lib.utils.cython_bbox import bbox_overlaps
import time


def divide_region_cy(regions):
    """ Divide the regions into non-overlapping sub-regions
        The algorithm first finds the shorter side of a region,
        it then divides the image along that axis into 2 parts.
        Then it finds the closest division along the longer axis
        so that the generated regions are close to squares
    """
    regions.astype(np.float, copy=False)
    return div.divide_region(regions, np.float(opt.SEAR_MIN_SIDE))


def _compute_ex_rois(size, gt_rois):
    """ Generate RoIs by zoom in to ideal grid (with random disturbances)
    """
    Bsel = np.zeros((0, 4))
    # the labels for zoom
    zoom_labels = np.zeros((0,))
    # the current set of regions
    w = size[1] - 1.0
    h = size[0] - 1.0
    lengths = np.array([[w, h, w, h]])
    for _ in range(opt.TRAIN_REP):  # 8
        # the current set of regions
        B = lengths * opt.TRAIN_ADDREGIONS
        # the set for zoom in
        Z = B
        # number of layers for the search
        height = size[0]
        width = size[1]
        side = np.minimum(height, width)
        K = int(np.log2(side / opt.SEAR_MIN_SIDE) + 1.0)
        for _ in range(K):
            # compute zoom labels
            zScores = _compute_zoom_labels(B, gt_rois)
            # selected regions
            Bsel = np.vstack((Bsel, B))
            # zoom labels
            zoom_labels = np.hstack((zoom_labels, zScores))
            # error vector
            err = (np.random.random(size=B.shape[0]) <=
                   opt.SEAR_ZOOM_ERR_PROB)
            # decide where to zoom
            indZ = np.where(np.logical_xor(zScores, err))[0]
            # Z is updated to regions for zoom in
            Z = B[indZ, :]
            if Z.shape[0] == 0:
                break
            # B is updated to be regions that are expanded from it
            # B = divide_region(Z, opt.SEAR_MIN_SIDE, opt.sub_divide, True)
            B = divide_region_cy(regions=Z)

        # add positive examples for sub-regions
    # print(Bsel.shape)
    for n in range(gt_rois.shape[0]):
        ri = gt_rois[n, :]
        # lengths of the RoI
        li = np.array([[ri[2] - ri[0] + 1.0, ri[3] - ri[1] + 1.0]])
        # scale lengths of the sub-region templates
        rt = np.array(opt.SUBREGION)
        lt = np.hstack((rt[:, [2]] - rt[:, [0]], rt[:, [3]] - rt[:, [1]]))
        # target super region lengths
        ls = li / lt
        # super region top-left corners
        ts = np.hstack((ri[0] - ls[:, [0]] * rt[:, [0]], ri[1] - ls[:, [1]] * rt[:, [1]]))
        # super region coordinates
        rs = np.hstack((ts, ts[:, [0]] + ls[:, [0]] - 1,
                        ts[:, [1]] + ls[:, [1]] - 1))

        Bsel = np.vstack((Bsel, rs))
        zoom_labels = \
            np.hstack((zoom_labels,
                       _compute_zoom_labels(rs, gt_rois)))

    # clip boxes to ensure they are within the boundary
    Bsel = _clip_boxes(Bsel, size)

    heights = Bsel[:, 3] - Bsel[:, 1] + 1
    widths = Bsel[:, 2] - Bsel[:, 0] + 1
    sides = np.minimum(heights, widths)
    keep_inds = np.where(sides >= opt.SEAR_MIN_SIDE)[0]

    return Bsel[keep_inds, :], zoom_labels[keep_inds]


def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes


def _compute_zoom_labels(rois, gt_rois):


    max_area_ratio = opt.max_area_ratio
    min_overlaps = opt.min_overlaps
    # overlaps, area_ratio = bbox_zoom_labels(rois, gt_rois, max_area_ratio)
    rois = at.tonumpy(rois)
    gt_rois = at.tonumpy(gt_rois)
    overlaps, area_ratio = bbox_zoom_labels_cy(rois.astype(np.float, copy=False), gt_rois.astype(np.float, copy=False), np.float(max_area_ratio))
    zoom_labels = np.any((area_ratio <= max_area_ratio) &
                        (overlaps >= min_overlaps), axis=1)

    return zoom_labels


def _compute_targets(gt_rois, ex_rois):
    """Compute bounding-box regression targets for an image.
            gt_rois: ground truth rois
            ex_rois: example rois
    """
    gt_rois = at.tonumpy(gt_rois)
    ex_rois = at.tonumpy(ex_rois)

    K = ex_rois.shape[0]
    N = gt_rois.shape[0]
    # Ensure ROIs are floats
    gt_rois = gt_rois.astype(np.float, copy=False)
    ex_rois = ex_rois.astype(np.float, copy=False)

    # bbox targets: (x1,y1,x2,y2,ex_rois_ind,subreg_ind)
    # subreg_ind , parant region is ex_roi
    targets = np.zeros((0, 7), dtype=np.float32)

    if K == 0 or N == 0:
        return targets

    # For each region, find out objects that are adjacent
    # Match objects to sub-regions with maximum overlaps.
    # Objects with large overlaps with any sub-regions are given priority.

    overlaps = bbox_overlaps(ex_rois, gt_rois)  # K,N
    max_overlaps = overlaps.max(axis=1)  # K, 1

    for k in range(K):
        if max_overlaps[k] < opt.SEAR_ADJ_THRESH:
            continue
        re = ex_rois[k, :]
        L = np.array([[re[2] - re[0], re[3] - re[1], re[2] - re[0], re[3] - re[1]]])
        delta = np.array([[re[0], re[1], re[0], re[1]]])
        # sub-regions`
        s_re = (L * opt.SUBREGION) + delta  # sub region coordinate  11, 4
        # compute the overlaps between sub-regions and each objects
        sre_gt_overlaps = bbox_overlaps(s_re, gt_rois)  # 11, N
        # find out the objects that are actually adjacent, find which  gt_rois they adjacent
        adj_th = (sre_gt_overlaps[0] >= opt.SEAR_ADJ_THRESH)  # the first row is themselves.
        match_inds = np.where(adj_th)[0]
        sre_gt_overlaps[:, ~adj_th] = -1
        #        adj_th = (sre_gt_overlaps >= cfg.SEAR.ADJ_THRESH)
        #        match_inds = np.where(np.any(adj_th, axis=0))[0]
        if match_inds.shape[0] > 0:  # there is object to match
            for _ in range(min(opt.NUM_SUBREG, match_inds.shape[0])):
                reg_idx, gt_idx = np.unravel_index(sre_gt_overlaps.argmax(),
                                                   sre_gt_overlaps.shape)

                # no more valid match
                #                if sre_gt_overlaps[reg_idx, gt_idx] < cfg.SEAR.ADJ_THRESH:
                #                    break
                t_ki = _compute_bbox_deltas(ex_rois[[k], :],
                                            gt_rois[[gt_idx], :])
                new_target = np.hstack((t_ki, np.array([[k, reg_idx, overlaps[k, gt_idx]]])))
                targets = np.vstack((targets, new_target))


                sre_gt_overlaps[reg_idx, :] = -1
                sre_gt_overlaps[:, gt_idx] = -1
    return targets

def _compute_bbox_deltas(ex, gt):
    ex_widths = np.maximum(ex[:, [2]] - ex[:, [0]], 1) + opt.EPS
    ex_heights = np.maximum(ex[:, [3]] - ex[:, [1]], 1) + opt.EPS
    ex_ctr_x = ex[:, [0]] + 0.5 * ex_widths
    ex_ctr_y = ex[:, [1]] + 0.5 * ex_heights

    gt_widths = np.maximum(gt[:, [2]] - gt[:, [0]], 1) + opt.EPS
    gt_heights = np.maximum(gt[:, [3]] - gt[:, [1]], 1) + opt.EPS
    gt_ctr_x = gt[:, [0]] + 0.5 * gt_widths
    gt_ctr_y = gt[:, [1]] + 0.5 * gt_heights

    ex_widths = np.maximum(1, ex_widths)
    ex_heights = np.maximum(1, ex_heights)
    gt_widths = np.maximum(1, gt_widths)
    gt_heights = np.maximum(1, gt_heights)

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    return np.hstack((targets_dx, targets_dy, targets_dw, targets_dh))


def _sample_rois(zoom_labels, rois, bbox_targets, fg_rois_per_image, rois_per_image):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    adj_labels = np.zeros((rois.shape[0], opt.NUM_SUBREG))
    adj_matching = bbox_targets[:, 4:6].astype(np.uint32, copy=False)  # reg_index, sub_region index
    IOU_target = bbox_targets[:, -1]  # overlaps
    for cls in range(opt.NUM_SUBREG):
        cls_inds = np.where(adj_matching[:, 1] == cls)[0]
        if not opt.SEAR_SCALE_ADJ_CONF:
            adj_labels[adj_matching[cls_inds, 0], cls] = 1
        else:
            adj_labels[adj_matching[cls_inds, 0], cls] = \
                IOU_target[cls_inds]

    # Find foreground RoIs
    fg_inds = np.where((adj_labels.any(axis=1) == 1) |
                       (zoom_labels == 1))[0]

    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = int(np.minimum(fg_rois_per_image, fg_inds.size))
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = np.random.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Find background RoIs
    bg_inds = np.where((adj_labels.any(axis=1) == 0) |
                       (zoom_labels == 0))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image,
                             replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sample values from various arrays
    adj_labels = adj_labels[keep_inds]  # n x 11
    zoom_labels = zoom_labels[keep_inds]  # n x 1

    rois_keep = rois[keep_inds]  # n x 4

    adj_targets, adj_loss_weights = \
        _get_adjacent_targets(bbox_targets,
                              keep_inds,
                              rois.shape[0],
                              opt.NUM_SUBREG)

    return adj_labels, zoom_labels, rois_keep, adj_targets, adj_loss_weights


def _get_adjacent_targets(compact_targets, keep_inds, num_regions, num_classes):
    """Get adjacent prediction labels
    """
    bbox_targets = np.zeros((num_regions, 4 * num_classes), dtype=np.float32)
    bbox_loss_weights = np.zeros((num_regions, 4 * num_classes), dtype=np.float32)
    # print(num_regions)

    for cls in range(num_classes):
        start = 4 * cls
        end = start + 4
        cls_inds = np.where(compact_targets[:, -2] == cls)[0]
        reg_inds = compact_targets[cls_inds, -3]
        for i in range(len(reg_inds)):
            ind = int(reg_inds[i])
            bbox_targets[ind, start:end] = compact_targets[cls_inds[i], 0:4]
            bbox_loss_weights[ind, start:end] = [1., 1., 1., 1.]

    return bbox_targets[keep_inds], bbox_loss_weights[keep_inds]


def generate_az_rois(size, gt_rois):
    """generate rois used for az_net training

    Args:
        size (2):  H x W of the origin images
        gt_rois (n, 4): ground rois

    """
    gt_rois = at.tonumpy(gt_rois)
    ex_rois, zoom_labels = _compute_ex_rois(size, gt_rois)
    targets = _compute_targets(gt_rois, ex_rois)

    adj_labels, zoom_labels, im_rois, adj_targets, adj_loss_weight \
        = _sample_rois(zoom_labels, ex_rois, targets, 32, 128)

    return adj_labels, zoom_labels, im_rois, adj_targets, adj_loss_weight



if __name__ == '__main__':
    from datasets.dataset import Dataset
    from torch.utils import data as data_
    from tqdm import tqdm

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, batch_size=1, \
                                  shuffle=True,
                                  num_workers=opt.num_workers)
    start = time.time()

    for i, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
        # scale = at.scalar(scale)
        img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
        size = img.shape[2:]
        bbox = at.tonumpy(bbox[0])
        # print('number of gt_boxes:', bbox.shape)
        ex_rois, zoom_labels = _compute_ex_rois(size, bbox)
        # print('ex_box:', ex_rois.shape)

        targets = _compute_targets(bbox, ex_rois)
        # print('targets.shape', targets.shape)
        adj_labels, zoom_labels, im_rois, adj_targets, adj_loss \
            = _sample_rois(zoom_labels, ex_rois, targets, 16, 32)

        print('adj_labels:', adj_labels.sum())

        # print(i,':','a:', a.shape,'b:', b.shape)

    b = time.time()
    print(b-start)
