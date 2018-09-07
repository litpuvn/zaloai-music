from __future__ import print_function 
import os
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision.models as models
import torch
import torch.nn as nn
import pickle
import pdb
import torch.optim as optim
from PIL import Image
import numpy as np
import random
import torch.backends.cudnn as cudnn
from time import time
import glob
import math

from inceptionresnetv2 import *

def get_fns_lbs(args,base_dirs,extension=".npy" ,pickle_fn = 'mydata.p', force = True):

    # pdb.set_trace()
    fns_a = []
    lbs_a = []
    cnt_a = 0
    for base_dir in base_dirs:
        this_pickle_fn = base_dir + pickle_fn
        if os.path.isfile(this_pickle_fn) and not force:
            mydata = pickle.load(open(this_pickle_fn, 'rb'))
            fns = mydata['fns']
            lbs = mydata['lbs']
            cnt = mydata['cnt']
            fns_a += fns
            lbs_a += lbs
            cnt_a += cnt
            continue
        fns = glob.glob("%s/*%s" % (base_dir, extension))
        def downsample_class_8(file):
            m_class = int(file.split("_")[-1].split(".")[0])
            part    = int(file.split("/")[-1].split("_")[0])
            re_sample = {1:2, 2: 1.7, 3:1.7, 4:1.5, 5:1.5, 6:1.55, 7:1.5, 8:0.6, 9:2, 10:2}
            
            if part < re_sample[m_class] * args.downsample:
                return True
            else:
                return False
        fns = list(filter(lambda x: downsample_class_8(x), fns))
        
        lbs = [int(name_file.split("_")[-1].split(".")[0]) - 1 for name_file in fns]
        
        cnt = len(fns)
        if cnt==0:
            continue
        
        mydata = {'fns':fns, 'lbs':lbs, 'cnt':cnt}
        pickle.dump(mydata, open(this_pickle_fn, 'wb'))
        print(os.path.isfile(this_pickle_fn))
        fns_a += fns
        lbs_a += lbs
        cnt_a += cnt

    return fns_a, lbs_a, cnt_a

class MyDataset(Dataset):

    def __init__(self, filenames, labels,transform=None):
        assert len(filenames) == len(labels), "Number of files != number of labels"
        self.fns = filenames
        self.lbs = labels
        self.transform = transform
        

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        image = Image.fromarray(np.load(self.fns[idx]))
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)        
        return image, self.lbs[idx], self.fns[idx]

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10,cut=1,dup=1):
        self.inplanes = dup*64//cut
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, dup*64//cut, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(dup*64//cut)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, dup*64//cut, layers[0])
        self.layer2 = self._make_layer(block, dup*128//cut, layers[1], stride=2)
        self.layer3 = self._make_layer(block, dup*256//cut, layers[2], stride=2)
        self.layer4 = self._make_layer(block, dup*512//cut, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(dup*512//cut * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def zaloresnet(depth,**kwargs):
    if depth == 8:
        model = ResNet(BasicBlock, [1,1,1,1], **kwargs)
    if depth == 18:
        model = ResNet(BasicBlock, [2,2,2,2], **kwargs)
    elif depth == 34:
        model = ResNet(BasicBlock, [3,4,6,3], **kwargs)
    elif depth == 50:
        model = ResNet(Bottleneck, [3,4,6,3], **kwargs)
    elif depth == 152:
        model = ResNet(Bottleneck, [3,8,36,3], **kwargs)
    return model
  
class ResNetBaseLine(nn.Module):
    def __init__(self, depth, num_classes, pretrained = True,dropout=0.3):
        super(ResNetBaseLine, self).__init__()
        if depth == 18:
            model = models.resnet18(pretrained)
        elif depth == 34:
            model = models.resnet34(pretrained)
        elif depth == 50:
            model = models.resnet50(pretrained)
        elif depth == 152:
            model = models.resnet152(pretrained)

        self.num_ftrs = model.fc.in_features
        # self.num_classes = num_classes

        self.shared = nn.Sequential(*list(model.children())[:-1])
        self.target = nn.Sequential(nn.Linear(self.num_ftrs, num_classes*10),\
                                    nn.ReLU(),\
                                    nn.Dropout(p=dropout),
                                    nn.Linear(num_classes*10,num_classes))

    def forward(self, x):
        # pdb.set_trace()

        x = self.shared(x)
        x = torch.squeeze(x)
        return self.target(x)

    def frozen_until(self, to_layer):
        print('Frozen shared part until %d-th layer, inclusive'%to_layer)

        # if to_layer = -1, frozen all
        child_counter = 0
        for child in self.shared.children():
            if child_counter <= to_layer:
                print("child ", child_counter, " was frozen")
                for param in child.parameters():
                    param.requires_grad = False
                # frozen deeper children? check
                # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
            else:
                print("child ", child_counter, " was not frozen")
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1

class InceptionResnetV2BaseLine(nn.Module):
    def __init__(self, depth,num_classes, pretrained = True,dropout=0.3):
        super(InceptionResnetV2BaseLine, self).__init__()
        
        self.model = inceptionresnetv2(num_classes=1000, pretrained='imagenet')

        self.num_ftrs = 1536
        # self.num_classes = num_classes
        
        self.shared = nn.Sequential(*list(self.model.children())[:-1])
        self.model.last_linear  = nn.Sequential(nn.Linear(self.num_ftrs, num_classes*10),\
                                    nn.ReLU(),\
                                    nn.Dropout(p=dropout),
                                    nn.Linear(num_classes*10,num_classes))

    def forward(self, input):
        x = self.model.features(input)
        x = self.model.logits(x)
        return x

    def frozen_until(self, to_layer):
        print('Frozen shared part until %d-th layer, inclusive'%to_layer)

        # if to_layer = -1, frozen all
        child_counter = 0
        for child in self.shared.children():
            if child_counter <= to_layer:
                print("child ", child_counter, " was frozen")
                for param in child.parameters():
                    param.requires_grad = False
                # frozen deeper children? check
                # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
            else:
                print("child ", child_counter, " was not frozen")
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1

            
            
##########################################################################
def mytopk(pred, gt, k=3):
    """
    compute topk error
    pred: (n_sample,n_class) np array
    gt: a list of ground truth
    --------
    return:
        n_correct: number of correct prediction 
        topk_error: error, = n_connect/len(gt)
    """
    # topk = np.argpartition(pred, -k)[:, -k:]
    topk = np.argsort(pred, axis = 1)[:, -k:][:, ::-1]
    diff = topk - np.array(gt).reshape((-1, 1))
    n_correct = np.where(diff == 0)[0].size 
    topk_error = float(n_correct)/pred.shape[0]
    return n_correct, topk_error


def net_frozen(args, model,isScratch):
    print('********************************************************')
    if not isScratch:
        model.frozen_until(args.frozen_until)
    init_lr = args.lr
    if args.net_type == 'resnet':
        if args.trainer.lower() == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                    lr=init_lr, weight_decay=args.weight_decay)
        elif args.trainer.lower() == 'sgd':
            layer_params = list(filter(lambda p: p.requires_grad, model.parameters()))
            
            optimizer = optim.SGD([{'params':layer_params[i],'lr':init_lr*(2*i+1)} for i in range(len(layer_params))], 
                    lr=init_lr,  weight_decay=args.weight_decay,momentum=0.9)
    else:
        if args.trainer.lower() == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                lr=init_lr, weight_decay=args.weight_decay)
        elif args.trainer.lower() == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                lr=init_lr,  weight_decay=args.weight_decay)
    print('********************************************************')
    return model, optimizer


def parallelize_model(model):
    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    return model


def unparallelize_model(model):
    try:
        while 1:
            # to avoid nested dataparallel problem
            model = model.module
    except AttributeError:
        pass
    return model


def second2str(second):
    h = int(second/3600.)
    second -= h*3600.
    m = int(second/60.)
    s = int(second - m*60)
    return "{:d}:{:02d}:{:02d} (s)".format(h, m, s)


def print_eta(t0, cur_iter, total_iter):
    """
    print estimated remaining time
    t0: beginning time
    cur_iter: current iteration
    total_iter: total iterations
    """
    time_so_far = time() - t0
    iter_done = cur_iter + 1
    iter_left = total_iter - cur_iter - 1
    second_left = time_so_far/float(iter_done) * iter_left
    s0 = 'Epoch: '+ str(cur_iter + 1) + '/' + str(total_iter) + ', time so far: ' \
        + second2str(time_so_far) + ', estimated time left: ' + second2str(second_left)
    print(s0)

def cvt_to_gpu(X):
    return Variable(X.cuda()) if torch.cuda.is_available() \
        else Variable(X)
