from __future__ import division
from zalo_utils import *
import torch
import torch.nn as nn
import argparse
import copy
import random
from torchvision import transforms
import torch.backends.cudnn as cudnn
import os, sys
from time import time, strftime
import  glob
import pandas as pd
parser = argparse.ArgumentParser(description='Zalo music classification Inference')


args = parser.parse_args()


data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomSizedCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize(mean, std)
        ]),
    'val': transforms.Compose([
        #transforms.Scale(224),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize(mean, std)
    ]),
}




data_dirs       = ['../train/fea_S_parts/',\
                  '../test/fea_S_parts/',\
                  '../val/fea_S_parts/',\
                  ]

def infer_from_dataset(path_dirs):
    fns = []
    
    for fol in path_dirs:
        fn = glob.glob("%s*npy" % (fol))
        fns = fns+ fn
    lbs = [-1]*len(fns)
    dsets = dict()
    dsets['test'] = MyDataset(fns, lbs, transform=data_transforms['val'])

    dset_loaders = {
        x: torch.utils.data.DataLoader(dsets[x],
                                       batch_size=256,
                                       shuffle= False,
                                       num_workers=5)
        for x in ['test']
    }

    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    
    for batch_idx, (inputs, labels, fns0) in enumerate(dset_loaders['test']):
        
        batch_mean = np.mean(inputs.numpy(), axis=(0,2,3))
        batch_std0 = np.std(inputs.numpy(), axis=(0,2,3))
        batch_std1 = np.std(inputs.numpy(), axis=(0,2,3), ddof=1)

        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)
        pop_std1.append(batch_std1)
        
    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std0 = np.array(pop_std0).mean(axis=0)
    pop_std1 = np.array(pop_std1).mean(axis=0)
    return pop_mean, pop_std0,pop_std1

print(infer_from_dataset(data_dirs))
