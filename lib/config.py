import numpy as np
from pprint import pprint


class Config:
    # data
    voc_data_dir = '/home/lz/Lab/pytorch/pytorch_az/datasets/tomato/'
    min_size = 400
    max_size = 800

    num_workers = 32
    n_class = 2


    sub_divide = np.array([[0, 0, 0.5, 0.5],
                           [0, 0.5, 0.5, 1],
                           [0.5, 0, 1, 0.5],
                           [0.5, 0.5, 1, 1],
                           [0.25, 0.25, 0.75, 0.75]])




    SEAR_MIN_SIDE = 10
    # zoom label
    max_area_ratio = 0.25
    min_overlaps = 0.5

    # Visualize
    env = 'ectractor_az'
    cls_env = 'classifier'
    plot_every = 10

    # 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
    DEDUP_BOXES = 1. / 16.



    # train
    az_lr = 0.001
    az_lr_decay = 0.2
    az_epoch = 1000

    az_train_num_proposals = 256
    az_pos_ratio = 0.4

    cls_lr = 0.001
    cls_lr_decay = 0.2
    cls_epoch = 1000

    BN = False

    # sigma for l1_smooth_loss
    az_sigma = 1.
    roi_sigma = 1.

    weight_decay = 0.0005
    # Overlap threshold for a ROI to be considered background (class = 0 if
    # overlap in [LO, HI))
    TRAIN_BG_THRESH_LO = 0.1
    TRAIN_BG_THRESH_HI = 0.5
    # Overlap required between a ROI and ground-truth box in order for that ROI to
    # be used as a bounding-box regression training example
    TRAIN_BBOX_THRESH=0.5

    # Fraction of minibatch that is labeled foreground (i.e. class > 0)
    TRAIN_FG_FRACTION = 0.1

    # Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
    TRAIN_FG_THRESH = 0.5

    TRAIN_REP = 8
    TRAIN_NUM_PROPOSALS = 2000
    TRAIN_ADDREGIONS = [[0, 0, 1, 1],
                        [0, 0, 0.8, 0.8],
                        [0, 0.2, 0.8, 1],
                        [0.2, 0, 1, 0.8],
                        [0.2, 0.2, 1, 1]]

    SEAR_ZOOM_ERR_PROB = 0.3
    SEAR_ADJ_THRESH = 0.1
    # batch size of region processing (to prevent excessive GPU memory consumption)
    SEAR_BATCH_SIZE = 5000
    SUBREGION = [[0, 0, 1, 1],
                 [-0.5, 0, 0.5, 1], [0.5, 0, 1.5, 1],
                 [0, -0.5, 1, 0.5], [0, 0.5, 1, 1.5],
                 [0, 0, 0.5, 1], [0.5, 0, 1, 1],
                 [0, 0, 1, 0.5], [0, 0.5, 1, 1],
                 [0.25, 0, 0.75, 1], [0, 0.25, 1, 0.75]]
    NUM_SUBREG = len(SUBREGION)
    SEAR_SCALE_ADJ_CONF = False

    TEST_NUM_PROPOSALS = 300

    # threshold in confidence score
    SEAR_Tc = 0.05
    SEAR_FIXED_PROPOSAL_NUM = True
    SEAR_NUM_PROPOSALS = 2000



    EPS = 1e-14
    vis_threshold = 0.5


    USE_GPU_NMS = True
    GPU_ID = 0
    Use_classifier_extractor = False

    # threshold at zoom indicator
    def Tz(self, mode, thresh=0.0):
        if mode == 'Train':
            return thresh
        else:
            threshold = 0.01
            return threshold



    def _parse(self, kwargs):
        print(kwargs)
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)  # setattr(x, 'y', v) is equivalent to ``x.y = v''

        print('************use config************')
        pprint(self._state_dict())

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}





















opt = Config()