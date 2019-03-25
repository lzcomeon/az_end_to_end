from __future__ import absolute_import
from __future__ import division
from lib.config import opt
import torch as t
import torch.nn as nn
from torch.nn import functional as F
from generic_model import GM
from model.az_data_layer.az_forward_bn import az_layer_bn
from model.extractor.VGG16 import vgg_16, resnet
from model.classifier.classifier_forward_bn import cls_layer_bn
from datasets.dataset import Dataset,inverse_normalize
from torch.utils import data as data_
import lib.array_tool as at
from lib.vis_tool import visdom_bbox




def train(**kwargs):
    opt._parse(kwargs)
    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=opt.num_workers)
    extractor = vgg_16(pretrain=True)
    az = az_layer_bn(1./16)
    classifier = cls_layer_bn(1./16)

    # load_path_az = '/home/lz/Lab/pytorch/pytorch_az/checkpoints/Extractor_AZ_BN/03141749.pkl_0.0746727'
    # load_az_dict = t.load(load_path_az)
    # extractor_dict = {k[10:]: v for k, v in load_az_dict.items() if k[:9] == 'extractor'}
    # ext_ = extractor.state_dict()
    # ext_.update(extractor_dict)
    # extractor.load_state_dict(ext_)
    # az_dict = {k[3:]: v for k, v in load_az_dict.items() if k[:2] == 'az'}
    # az_ = az.state_dict()
    # az_.update(az_dict)
    # az.load_state_dict(az_)
    #
    # load_path_classifier = '/home/lz/Lab/pytorch/pytorch_az/checkpoints/Classifier_AZ_BN/03140602.pkl_0.027193144'
    # load_classifier_dict = t.load(load_path_classifier)
    # classifier_dict = {k[11:]: v for k, v in load_classifier_dict.items() if k[:10] == 'classifier'}
    # classifier_ = classifier.state_dict()
    # classifier_.update(classifier_dict)
    # classifier.load_state_dict(classifier_)

    trainer = GM(extractor, az, classifier).cuda()
    # load_path = '/home/lz/Lab/pytorch/pytorch_az/checkpoints/end_to_end/03191806.pkl_0.27578926'
    # load_dict = t.load(load_path)
    # trainer.load_state_dict(load_dict)



    lr_ = opt.cls_lr
    learn_de = 1

    for epoch in range(500):
        for ii, (img, bbox_, label_, scale) in enumerate(dataloader):
            # scale = at.scalar(scale)
            # continue
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            bbox = bbox[0]
            label = label[0]
            losses = trainer.train_step(img, bbox, label, scale)
            if (ii+1) % opt.plot_every == 0:
                # trainer.vis.log(str(losses))
                trainer.vis.plot_many(trainer.get_meter_data())
                current_loss = []
                for k, v in losses._asdict().items():
                    current_loss.append(v.detach().cpu().numpy())
                print('Epoch:', epoch, 'batch:', ii, 'zoom:', current_loss[0], \
                      'az_cls:', current_loss[1],'az_bbox:', current_loss[2], 'cls', current_loss[3],
                'loc', current_loss[4],'loss', current_loss[5],'lr:', lr_)

                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)

                # plot predict bboxes
                _bboxes, _labels, _scores = trainer.predict(img, visualize=True)
                pred_img = visdom_bbox(at.tonumpy(ori_img_),
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)





        if (epoch + 1) % pow(4, learn_de) == 0:
            trainer.scale_lr(opt.az_lr_decay)
            learn_de += 1

        lr_ = trainer.optimizer.param_groups[0]['lr']

        if (epoch+1) % 8 == 0:
            total_loss = losses.total_loss.detach().cpu().numpy()
            trainer.save(loss=total_loss, save_path=None)

            # print('train ok')



if __name__ == '__main__':
    import fire

    fire.Fire()












