from __future__ import absolute_import
from __future__ import division
from lib.config import opt
import lib.utils.cython_div as div
import numpy as np
import torch as t
import lib.array_tool as at



def im_propose(net, features, size, num_proposals = opt.SEAR_NUM_PROPOSALS):
    """Generate object proposals using AZ-Net
    Arguments:
        net(pytorch model): AZ-Net model
        im (ndarry): color image to test
    Returns:
        Y: R x 5 array of proposal
    """
    # the set for zoom in
    # Z = np.vstack((B, divide_region(B)))
    # the set of region proposals from adjacent predictions
    Y = np.zeros((0, 4))
    # confidence scores of adjacent predictions
    aScores = np.zeros((0,))
    zScores =np.zeros((0,))
    bbox_info = np.zeros((0, 10))
    rois = np.zeros((0, 4))
    # number of evaluations
    num_eval = 0
    # number of layers for the search
    height = size[1]
    width = size[0]
    side = np.minimum(height, width)
    K = int(np.log2(side / opt.SEAR_MIN_SIDE) + 1.)
    # the current set of regions
    l = np.array([[width-1, height-1, width-1, height-1]])
    B = l*opt.TRAIN_ADDREGIONS

    Tz = opt.Tz(mode='Train')
    # conv = None

    for k in range(1, K):
        zoom, score, bbox, info = _az_forward(net, features, B, size)
        num_eval = num_eval + B.shape[0]
        Y = np.vstack((Y, bbox))
        bbox_info = np.vstack((bbox_info, info))
        aScores = np.hstack((aScores, score))
        zScores = np.hstack((zScores, zoom))
        if k == 1:
            zoom[0] = 1
        indZ = np.where(zoom > Tz)[0]
        Z = B[indZ, :]
        if Z.shape == 0:
            break
        B = divide_region(Z)

    indA = np.argsort(-aScores)
    max_num = np.minimum(num_proposals, Y.shape[0])
    Y = Y[indA[:max_num], :]
    A = aScores[indA[:max_num]]
    # Info = bbox_info[indA[:max_num], :]

    return Y, A






def _az_forward(net, features, all_boxes, size, conv=None):
    """

    :param net:
    :param im:
    :param all_boxes:
    :param conv:
    :return:
    """
    batch_size = opt.SEAR_BATCH_SIZE
    num_batches = int(np.ceil(all_boxes.shape[0] / float(batch_size)))

    zScores = np.zeros((0,))
    aBBox = np.zeros((0, 4))
    cScores = np.zeros((0,))
    box_info = np.zeros((0, 10))

    for bid in range(num_batches):
        start = batch_size * bid
        end = min(all_boxes.shape[0], batch_size*(bid + 1))
        boxes = all_boxes[start: end, 0:4]


        if opt.DEDUP_BOXES > 0:
            v = np.array([1e3, 1e6, 1e9, 1e12])
            hashes = np.round(boxes * opt.DEDUP_BOXES).dot(v)
            new, index, inv_index = np.unique(hashes, return_index=True,
                                            return_inverse=True)
            boxes = boxes[index, :]

        zoom, scores, bbox_deltas = net(features, at.totensor(boxes))
        zoom = at.tonumpy(zoom)
        scores = at.tonumpy(scores)
        bbox_deltas = at.tonumpy(bbox_deltas)  # n, 44
        pred_boxes = _bbox_pred(boxes, bbox_deltas)  # n, 44
        pred_boxes = _clip_boxes(pred_boxes, size)  # n, 44

        if opt.DEDUP_BOXES > 0:
            pred_scores = scores[inv_index, :]  # n, 11
            pred_boxes = pred_boxes[inv_index, :]  # n, 44
            z_tb = zoom[inv_index].ravel()
            deltas = bbox_deltas[inv_index, :]  # n, 44
            boxes = boxes[inv_index, :]



        pr_boxes, pr_boxes_sco, deltas_, ori_roi_index, anchor_index= \
            _unwrap_adj_pred(pred_boxes, pred_scores, deltas)

        # deltas, original bbox, zoom for original bbox, anchor index for pred bbox
        # 4+4+1+1 = 10
        deltas_ori_anchor_index = np.hstack((deltas_, boxes[ori_roi_index, :],
                                             z_tb[ori_roi_index][:, np.newaxis],
                                             anchor_index[:, np.newaxis]))

        zScores = np.hstack((zScores, z_tb))
        aBBox = np.vstack((aBBox, pr_boxes))
        cScores = np.hstack((cScores, pr_boxes_sco))
        box_info = np.vstack((box_info, deltas_ori_anchor_index))

    return zScores, cScores, aBBox, box_info




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
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], size[1])
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], size[0])

    return boxes

def _unwrap_adj_pred(boxes, scores, deltas):
    num_anchor = boxes.shape[1]/4
    scores = scores.ravel()
    boxes = boxes.reshape(-1, 4)
    deltas = deltas.reshape(-1, 4)
    heights = boxes[:, 3] - boxes[:, 1] + 1.
    widths = boxes[:, 2] - boxes[:, 0] + 1.

    sides = np.minimum(heights, widths)
    keep_inds = np.where(sides >= opt.SEAR_MIN_SIDE)[0]
    ori_roi_inds = np.trunc(keep_inds / num_anchor).astype(np.int)
    anchor_index = (keep_inds % num_anchor).astype(np.int)


    return boxes[keep_inds, :], scores[keep_inds], deltas[keep_inds, :], ori_roi_inds, anchor_index

def divide_region(regions):
    """ Divide the regions into non-overlapping sub-regions
        The algorithm first finds the shorter side of a region,
        it then divides the image along that axis into 2 parts.
        Then it finds the closest division along the longer axis
        so that the generated regions are close to squares
    """
    regions.astype(np.float, copy=False)
    return div.divide_region(regions, np.float(opt.SEAR_MIN_SIDE))


if __name__ == '__main__':
    model_path = '/home/lz/Lab/pytorch/pytorch_az/checkpoints/Extractor_AZ_BN/03130808.pkl_0.08616127'
    from datasets.dataset import Dataset, inverse_normalize
    import lib.array_tool as at
    from torch.utils import data as data_

    from model.extractor.VGG16 import vgg_16
    from model.az_data_layer.az_forward_bn import az_layer_bn

    extractor = vgg_16(pretrain=False).cuda()
    az = az_layer_bn(1.16).cuda()


    param = t.load(model_path)
    extractor_dict = {k[10:]: v for k, v in param.items() if k[:9] == 'extractor'}
    az_dict = {k[3:]: v for k, v in param.items() if k[:2] == 'az'}

    extractor.load_state_dict(extractor_dict)
    az.load_state_dict(az_dict)

    print(' all model loaded')


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
        features = extractor(img)

        aScore, regions, bbox_info = im_propose(az, features, size=img.shape[2:])
        print('regions', regions.shape, 'aScore', aScore.shape, bbox_info.shape)  # (2000,4)

