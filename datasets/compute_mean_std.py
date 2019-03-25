import os
import numpy as np
from PIL import Image
import threading



path = './tomato/JPEGImages/'

imgs = np.zeros((0, 3, 200000))
file = os.listdir(path)
means, stdevs = [], []
def work(i):
    imgs_th = np.zeros((0, 3, 200000))
    for j in range(i*100, (i+1)*100):
        name = file[j]
        dir = os.path.join(path, name)
        f = Image.open(dir)
        img = f.convert('RGB')
        img = np.asarray(img, dtype=np.float32)
        if hasattr(f, 'close'):
            f.close()
        data = img.transpose((2, 0, 1))
        data = data/255.
        # print(data.dtype)
        data_ = data.reshape(1, 3, -1, )
        imgs_th = np.vstack((imgs_th, data_))
    print(str(i*100), '~', str((i+1)*100), ' completed')
    return imgs_th


from threading import Thread

class MyThread(Thread):

    def __init__(self, number):
        Thread.__init__(self)
        self.number = number

    def run(self):
        self.result = work(self.number)

    def get_result(self):
        return self.result



ls = []

img_data = []




#
if __name__ =='__main__':
    for i in range(8):
        ls.append(MyThread(i))

    for i in ls:
        i.start()
        i.join()
        img_data.append(i.get_result())

    for data in img_data:
        print(data.shape)
        imgs = np.vstack((imgs, data))


    for i in range(3):
        pixels = imgs[:, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))










