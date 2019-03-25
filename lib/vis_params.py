import torch as t
from torch import nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np



def load_net(file):
    params = t.load(file)
    for k, v in dict(params):
        if k =='az.fc7_2.weight':
            pass

    print(params)




def plot_histogram():
    pass



def main():
    file = '/home/lz/Lab/pytorch/pytorch_az/checkpoints/Extractor_AZ_BN/03021147.pkl_0.245729'
    load_net(file)


if __name__ == '__main__':
    main()