from datasets.dataset import Dataset
from lib.config import opt
from torch.utils import data as data_
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import torch as t


def inverse_normalize(img):
    img = img.cpu().numpy()
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def test():

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, batch_size=1, \
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    for i, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
        # scale = at.scalar(scale)
        img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
        if i==1:
            print('img.shape:', img.shape)
            print('min:', img.min())
            print('max:', img.max())
            print('label:',label)
            print('bbox.shape:', bbox.shape)
            print(bbox)

            # inv_img = inverse_normalize(img)
            # print('inverse:', inv_img.shape)
            # print('min:', inv_img.min())
            # print('max:', inv_img.max())
            # inv_img = ToPILImage()(t.from_numpy(inv_img[0]))
            # inv_img.show()

            #
            # plt.imshow(inv_img)
            # plt.show()
        # break
        # print(roi_cls.shape)




def vis():
    path = './tomato/JPEGImages/001_1.jpg'
    from PIL import Image
    import numpy as np
    img = Image.open(path)

    # img = img.convert('RGB')
    # img = np.asarray(img, dtype=np.float32)
    plt.imshow(img)
    plt.show()








test()


# vis()