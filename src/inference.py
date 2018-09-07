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
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=18, choices = [8,18,34, 50, 152], type=int, help='depth of model')
parser.add_argument('--model_path', type=str, default = ' ')
parser.add_argument('--batch_size', type=str, default = 256)
parser.add_argument('--n_model',type=str,default = 'fea_S')
parser.add_argument('--num_workers', default=5, type=int)
parser.add_argument('--ver', default=1, type=int)

parser.add_argument('--inf_folder', default="test", type=str,
        help = "is scaling data to 224 or not?")

args = parser.parse_args()

KTOP = 3 # top k error

if args.net_type == 'resnet': 
    
    mean=[0.485, 0.456, 0.406]
    std= [0.229, 0.224, 0.225]
else:
    
    mean=[0.5, 0.5, 0.5]
    std= [0.5, 0.5, 0.5]
data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomSizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ]),
    'val': transforms.Compose([
        #transforms.Scale(224),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

#################
# load model 
old_model = './checkpoint/' + '%s_' % (args.n_model) + args.net_type + '-%s' % (args.depth) + '_' + args.model_path + '.t7'
print("| Load pretrained at  %s..." % old_model)
checkpoint = torch.load(old_model, map_location=lambda storage, loc: storage)
model = checkpoint['model']
model = unparallelize_model(model)
model = parallelize_model(model)
best_top1 = checkpoint['top1']
print('previous top1\t%.4f'% best_top1)
print('=============================================')

torch.set_grad_enabled(False)
model.eval()
k = 3 # top 3 error


data_dir_parts = '../%s/%s_parts/' % (args.inf_folder,args.n_model)


def infer_from_dataset(path_dir, part=None):
    tot = 0
    if part==None:
        fns = glob.glob(path_dir+"%s*npy" % str_scale)
        lbs = [-1]*len(fns)
    else:
        fns = glob.glob(path_dir + "%d_*npy" % (part))
        lbs = [-1] * len(fns)
     
    dsets = dict()
    dsets['test'] = MyDataset(fns, lbs, transform=data_transforms['val'])

    dset_loaders = {
        x: torch.utils.data.DataLoader(dsets[x],
                                       batch_size=args.batch_size,
                                       shuffle= False,
                                       num_workers=args.num_workers)
        for x in ['test']
    }
    df = pd.DataFrame()
    df_raw = pd.DataFrame()
    for batch_idx, (inputs, labels, fns0) in enumerate(dset_loaders['test']):
        
        inputs = cvt_to_gpu(inputs)
        outputs = model(inputs)
        outputs_softmax=torch.nn.functional.softmax(outputs).data.cpu().numpy()
        outputs = outputs.data.cpu().numpy()
        outputs_classes = np.argsort(outputs, axis = 1)[:, -k:][:, ::-1] + 1

        tot += len(fns0)
        
        if part==None:
            inputs = np.array([n_file.split("-")[1].split("_")[0] for n_file in fns0])
        else:
            if args.inf_folder == "chaval":
                inputs = np.array(["_".join(n_file.split("/")[-1].split("_")[1:-2]) for n_file in fns0])
            else:
                inputs = np.array([n_file.split("_")[-3] for n_file in fns0])
        
        inputs = inputs.reshape(len(inputs),-1)
        df     = pd.concat([df, pd.DataFrame(np.concatenate([inputs,outputs_classes],axis=1))])
        df_raw = pd.concat([df_raw, pd.DataFrame(np.concatenate([inputs,outputs_softmax],axis=1))])
        print('processed {}/{}'.format(tot, len(fns)))

    df_top=df.iloc[:,[0,1]] 
    if part==None:
        df_top.columns = ["Id","%s_f" % (args.n_model)]
        df.columns     = ["Id"] + ["%s_top%d_f" % (args.n_model,t+1) for t in range(k)]
        df_raw.columns = ["Id"] + ["%s_class%d_f" % (args.n_model,c+1) for c in range(outputs.shape[-1])]
    else:
        df_top.columns = ["Id","%s_%d" % (args.n_model,part)]
        df.columns = ["Id"] + ["%s_top%d_%d" % (args.n_model,t+1, part) for t in range(k)]
        df_raw.columns = ["Id"] + ["%s_class%d_%d" % (args.n_model,c+1,part) for c in range(outputs.shape[-1])]    

    return (df_top.set_index("Id").sort_index(), df.set_index("Id").sort_index(), df_raw.set_index("Id").sort_index())

dfs_top = []
dfs     = []
dfs_raw = []
'''
df_top, df,df_raw = infer_from_dataset(data_dir)
dfs_top.append(df_top)
dfs.append(df)
dfs_raw.append(df_raw)
'''
for part in range(8):
    df_top, df,df_raw = infer_from_dataset(data_dir_parts,part=part)
    dfs_top.append(df_top)
    dfs.append(df)
    dfs_raw.append(df_raw)

for df in dfs:
    print (df.head())

df = pd.concat(dfs,axis=1)
df.to_csv("../features_%s_%s_%d.csv" % (args.n_model, args.inf_folder,args.ver))

df_raw = pd.concat(dfs_raw,axis=1)
df_raw.to_csv("../features_raw_%s_%s_%d.csv" % (args.n_model, args.inf_folder,args.ver))

df_top = pd.concat(dfs_top,axis=1)
df_top.to_csv("../top_%s_%s_%d.csv" % (args.n_model, args.inf_folder,args.ver))

def vote(x):
    return x.groupby(x).count().idxmax()

df_top = df_top.apply(lambda x: vote(x),axis=1)
df_top = df_top.reset_index()
df_top.columns = ["Id","Genre"]
df_top.to_csv("../submission_%s_%s_%d.csv" % (args.n_model,args.inf_folder,args.ver),header=True,index=False)

print('done')
