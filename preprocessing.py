from __future__ import division

import torch
import torch.nn as nn
import argparse
import copy
import random
from torchvision import transforms
import torch.backends.cudnn as cudnn
import os, sys
import librosa
import  glob
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from matplotlib import cm

parser = argparse.ArgumentParser(description='Zalo preprocessing')
parser.add_argument('--num_workers', default=5, type=int, help = "number of workers")
parser.add_argument('--folder', default="test", type=str, help = "data select point")
parser.add_argument('--type', default="test", type=str, help = "data select point")

parser.add_argument('--fea', default="S",   type=str, help = "feature")
parser.add_argument('--batch_size', type=str, default = 1)
parser.add_argument('--shape', default="224",  choices=[224,299], type=int, help = "input shape")
parser.add_argument('--upsample', default=1, type=int, help="upsample to create a lot data")

args = parser.parse_args()

input_size = np.array([args.shape,args.shape])


import os
directory = args.type

if not os.path.exists(directory):
    os.makedirs(directory)

directory = "%s/%s/%s" % ("/model",args.type,"fea_" + args.fea)

if not os.path.exists(directory):
    os.makedirs(directory)

directory = "%s/%s/%s" % ("/model",args.type,"fea_" + args.fea+"_parts")

if not os.path.exists(directory):
    os.makedirs(directory)
def normalize(arr,min_,max_):
    return (arr-min_)/(max_ - min_)


class MyDataset(Dataset):

    def __init__(self, filenames, labels):
        assert len(filenames) == len(labels), "Number of files != number of labels"
        self.fns = filenames
        self.lbs = labels
        self.num_samples = {1:12,2:10,3:10,4:7,5:5,6:5,7:10,8:10,9:10,10:12,-1:3}
        self.num_first_components = [(input_size[0]*2)//2,\
                         	     (input_size[0]*3)//2,\
			             (input_size[0]*4)//2,\
				     (input_size[0]*5)//2,\
				     (input_size[0]*6)//2]

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        y, sr = librosa.load("%s/%s" % (args.folder,self.fns[idx]),sr=None)
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=100)
        # Convert to log scale (dB). We'll use the peak power (max) as reference.
        log_S = librosa.power_to_db(S, ref=np.max)
        np.save("%s/S_%s_%d.npy" % (args.type,self.fns[idx],self.lbs[idx]),log_S)
        im = Image.fromarray(np.uint8(cm.seismic(normalize(log_S,-80,0))*255))
        arr = np.asarray(im.resize((input_size[0]*4, input_size[-1]),Image.ANTIALIAS))
        np.random.seed(100)
        indexes = self.num_first_components + \
                   list(sorted(np.random.choice(range(input_size[0]//3,input_size[0]*3,1),\
                   self.num_samples[self.lbs[idx]]*args.upsample,replace=False)))
        
        for step in range(len(indexes)):
            from_ = indexes[step]
            save_file = args.type+"/fea_%s_parts/%d_%s_%s_%d.npy" %\
                                                  (args.fea,step,self.fns[idx],args.fea,self.lbs[idx])
            s_arr = arr[:,from_:from_+input_size[-1]] 
            np.save(save_file,s_arr)
        
        return self.fns[idx],self.lbs[idx]
    
data_dir_parts = '%s/' % (args.type)

if args.type == "train" or args.type == "val":
    df  = pd.read_csv("%s.csv" % args.type,header=None,names=["file",'class'])
    fns = df["file"]
    lbs = df["class"].astype(int)
else:
    try:
        df  = pd.read_csv("%s.csv" % args.type,header=None,names=["file"])
    except:
        fns = glob.glob("%s/*mp3" % (args.folder))
        fns = [f.split("/")[-1] for f in fns]

    lbs = [-1] * len(fns)


df = pd.DataFrame({"id":fns})
df.to_csv("test.csv",header=None,index=False)


dset = MyDataset(fns,lbs)

dset_loader = torch.utils.data.DataLoader(dset,
                                       batch_size=args.batch_size,
                                       shuffle= False,
                                       num_workers=args.num_workers)
print ("converting mp3 files to images is doing .... plz wait!")
for batch_idx, (fns0,lb) in enumerate(dset_loader):
    print (fns0,lb)
