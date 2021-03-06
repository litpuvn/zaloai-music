from __future__ import division
from zalo_utils import *
from sklearn.model_selection import train_test_split
# from utils import *
import torch
import torch.nn as nn
import argparse
import copy
import random
from torchvision import transforms
# import time
import torch.backends.cudnn as cudnn
import os, sys
from time import time, strftime

parser = argparse.ArgumentParser(description='PyTorch Zalo music classification')
parser.add_argument('--n_model', default="fea_S", type=str, help='select features')
parser.add_argument('--lr', default=5e-2, type=float, help='learning rate')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=18, choices = [8,18, 34, 50, 152], type=int, help='depth of model')
parser.add_argument('--weight_decay', default=5e-6, type=float, help='weight decay')
parser.add_argument('--weight_schedule', default=0, type=int, help='weight schedule')

parser.add_argument('--trainer', default='adam', type = str, help = 'optimizer')
parser.add_argument('--model_path', type=str, default = ' ')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--num_workers', default=15, type=int)
parser.add_argument('--num_epochs', default=1000, type=int,
                    help='Number of epochs in training')
parser.add_argument('--dropout_keep_prob', default=0.3, type=float)
parser.add_argument('--check_after', default=1,
                    type=int, help='check the network after check_after epoch')
parser.add_argument('--train_from', default=1,
                    choices=[0, 1, 2],  # 0: from scratch, 1: from pretrained Resnet, 2: specific checkpoint in model_path
                    type=int,
                    help="training from beginning (1) or from the most recent ckpt (0)")
parser.add_argument('--frozen_until', '-fu', type=int, default = 8,
                    help="freeze until --frozen_util block")
parser.add_argument('--downsample', default=100, type=int, 
        help = "limit frame of class 8")

parser.add_argument('--l1', default=1e-4, type=float, help='l1_regularization')
parser.add_argument('--l2', default=1e-4, type=float, help='l2_regularization')
parser.add_argument('--cut', default=1, type=int, help='reduce the capacity of model')
parser.add_argument('--dup', default=1, type=int, help='double the capacity of model')
args = parser.parse_args()


is_scale = True

print (args)


KTOP = 3 # top k error

def exp_lr_scheduler(args, optimizer, epoch):
    # after epoch 100, not more learning rate decay
    init_lr = args.lr
    lr_decay_epoch = 50 # decay lr after each 50 epoch
    weight_epoch = 10 # decay lr after each 50 epoch
    if args.weight_schedule == 1:
        weight_decay = args.weight_decay * (1.1 ** (min(epoch, 200) // weight_epoch))
    else:
        weight_decay = args.weight_decay
    
    lr = init_lr * (0.6 ** (min(epoch, 200) // lr_decay_epoch)) 

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = weight_decay

    return optimizer, lr, weight_decay


use_gpu = torch.cuda.is_available()

data_dirs_train =['../train/%s_parts/' % args.n_model] # + ['../val/%s_parts/' % args.n_model]
data_dirs_val   =['../val/%s_parts/' % args.n_model]

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

print('Loading data')

train_fns, train_lbs, cnt = get_fns_lbs(args,data_dirs_train)

print('Total train files in the original dataset: {}'.format(cnt))
print('Total train files with > 0 byes: {}'.format(len(train_fns)))
print('Total train files with zero bytes {}'.format(cnt - len(train_fns)))

val_fns, val_lbs, cnt = get_fns_lbs(args,data_dirs_val)
    
print('Total val files in the original dataset: {}'.format(cnt))
print('Total val files with > 0 byes: {}'.format(len(val_fns)))
print('Total val files with zero bytes {}'.format(cnt - len(val_fns)))


########### 
print('DataLoader ....')


dsets = dict()
dsets['train'] = MyDataset(train_fns, train_lbs, transform=data_transforms['train'])
dsets['val']   = MyDataset(val_fns, val_lbs, transform=data_transforms['val'])

dset_loaders = {
    x: torch.utils.data.DataLoader(dsets[x],
                                   batch_size=args.batch_size,
                                   shuffle=(x != 'val'),
                                   num_workers=args.num_workers)
    for x in ['train', 'val']
}
########## 
print('Load model')


if args.net_type == 'resnet': 
    
    class_model = ResNetBaseLine
else:
    
    class_model = InceptionResnetV2BaseLine


saved_model_fn = '%s_' % (args.n_model) +args.net_type + '-%s' % (args.depth) + '_' + strftime('%m%d_%H%M')
old_model = './checkpoint/' + '%s_' % (args.n_model) + args.net_type + '-%s' % (args.depth) + '_' + args.model_path + '.t7'
if args.train_from == 2 and os.path.isfile(old_model):
    print("| Load pretrained at  %s..." % old_model)
    checkpoint = torch.load(old_model, map_location=lambda storage, loc: storage)
    tmp = checkpoint['model']
    model = unparallelize_model(tmp)
    best_top3 = checkpoint['top3']
    print('previous top3\t%.4f'% best_top3)
    print('=============================================')
elif args.train_from == 1:
       
    model = class_model(args.depth, len(set(train_lbs)),dropout=args.dropout_keep_prob)
    
else:
    if args.net_type == 'resnet': 
        model = zaloresnet(args.depth,num_classes=len(set(train_lbs)),cut=args.cut,dup=args.dup)
    else:
        print "don't have cratch vgg model"
        assert False
##################
print('Start training ... ')
criterion = nn.CrossEntropyLoss()
model, optimizer = net_frozen(args, model,isScratch=(args.train_from == 0))
model = parallelize_model(model)

N_train = len(train_lbs)
N_valid = len(val_lbs)
best_top3 = 1
best_top1 = 1
best_top1_train = 1
best_top3_train = 1
t0 = time()
penanty_factor = 1

for epoch in range(args.num_epochs):
    optimizer, lr, weight_delay = exp_lr_scheduler(args, optimizer, epoch) 
    print('#################################################################')
    print('=> Training Epoch #%d, LR=%.10f, weight_delay=%.10f' % (epoch + 1, lr,weight_delay))
    # torch.set_grad_enabled(True)

    running_loss, running_corrects, tot = 0.0, 0.0, 0.0
    running_loss_src, running_corrects_src, tot_src = 0.0, 0.0, 0.0
    runnning_topk_corrects = 0.0
    ########################
    model.train()
    torch.set_grad_enabled(True)
    ## Training 
    # local_src_data = None
    for batch_idx, (inputs, labels, _) in enumerate(dset_loaders['train']):
        
        optimizer.zero_grad()
        inputs = cvt_to_gpu(inputs)
        labels = cvt_to_gpu(labels)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        l1_regularization  = penanty_factor*args.l1*torch.norm(torch.cat([x.view(-1) for x in model.parameters()]),1)
        l2_regularization  = penanty_factor*args.l1*torch.norm(torch.cat([x.view(-1) for x in model.parameters()]),1)
        loss+=(l1_regularization+l2_regularization)
        running_loss += loss*inputs.shape[0]
        loss.backward()
        optimizer.step()
        ############################################
        _, preds = torch.max(outputs.data, 1)
        # topk 
        top3correct, _ = mytopk(outputs.data.cpu().numpy(), labels, KTOP)
        runnning_topk_corrects += top3correct
        # pdb.set_trace()
        running_loss += loss.item()
        running_corrects += preds.eq(labels.data).cpu().sum()
        tot += labels.size(0)
        sys.stdout.write('\r')
        try:
            batch_loss = loss.item()
        except NameError:
            batch_loss = 0

        top1error = 1 - float(running_corrects)/tot
        top3error = 1 - float(runnning_topk_corrects)/tot
        sys.stdout.write('| Epoch [%2d/%2d] Iter [%3d/%3d]\tBatch loss %.4f\tTop1error %.4f \tTop3error %.4f'
                         % (epoch + 1, args.num_epochs, batch_idx + 1,
                            (len(train_fns) // args.batch_size), batch_loss/args.batch_size,
                            top1error, top3error))
        sys.stdout.flush()
        sys.stdout.write('\r')

    top1error = 1 - float(running_corrects)/N_train
    top3error = 1 - float(runnning_topk_corrects)/N_train
    epoch_loss = running_loss/N_train
    print('\n| Training loss %.4f\tTop1error %.4f \tTop3error: %.4f'\
            % (epoch_loss, top1error, top3error))

    print_eta(t0, epoch, args.num_epochs)

    top1error_train = top1error
    top3error_train = top3error

    #train_top1error = top1error
    ###################################
    ## Validation
    if (epoch + 1) % args.check_after == 0:
        # Validation 
        running_loss, running_corrects, tot = 0.0, 0.0, 0.0
        runnning_topk_corrects = 0
        torch.set_grad_enabled(False)
        model.eval()
        for batch_idx, (inputs, labels, _) in enumerate(dset_loaders['val']):
            inputs = cvt_to_gpu(inputs)
            labels = cvt_to_gpu(labels)
            outputs = model(inputs)
            _, preds  = torch.max(outputs.data, 1)
            top3correct, top3error = mytopk(outputs.data.cpu().numpy(), labels, KTOP)
            runnning_topk_corrects += top3correct
            running_loss += loss.item()
            running_corrects += preds.eq(labels.data).cpu().sum()
            tot += labels.size(0)

        epoch_loss = running_loss / N_valid 
        top1error = 1 - float(running_corrects)/N_valid
        top3error = 1 - float(runnning_topk_corrects)/N_valid
        #penanty_factor = (top1error/train_top1error)**1.5
        print('| Validation loss %.4f\tTop1error %.4f \tTop3error: %.4f \tpenanty_factor:%.4f'\
                % (epoch_loss, top1error, top3error,penanty_factor))
        

        ################### save model based on best top1 error
        if top1error  < best_top1 :
            print('Saving model')
            best_top1 = top1error
            best_model = copy.deepcopy(model)
            state = {
                'model': best_model,
                'top1' : best_top1,
                'args': args
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_point = './checkpoint/'
            if not os.path.isdir(save_point):
                os.mkdir(save_point)

            torch.save(state, save_point + saved_model_fn + '.t7')
            print('=======================================================================')
            print('model saved to %s' % (save_point + saved_model_fn + '.t7'))

         ################### save model based on best top1 error
        if top1error_train  < best_top1_train :
            print('Saving model')
            best_top1_train = top1error_train
            best_model = copy.deepcopy(model)
            state = {
                'model': best_model,
                'top1' : best_top1_train,
                'args': args
            }
            if not os.path.isdir('checkpoint_train'):
                os.mkdir('checkpoint_train')
            save_point = './checkpoint_train/'
            if not os.path.isdir(save_point):
                os.mkdir(save_point)

            torch.save(state, save_point + saved_model_fn + '.t7')
            print('=======================================================================')
            print('model saved to %s' % (save_point + saved_model_fn + '.t7'))

        ################### save model based on best top3 error
        if top3error  < best_top3 :
            print('Saving model')
            best_top3 = top3error
            best_model = copy.deepcopy(model)
            state = {
                'model': best_model,
                'top3' : best_top3,
                'args': args
            }
            if not os.path.isdir('checkpoint_top3'):
                os.mkdir('checkpoint_top3')
            save_point = './checkpoint_top3/'
            if not os.path.isdir(save_point):
                os.mkdir(save_point)

            torch.save(state, save_point + saved_model_fn + '.t7')
            print('=======================================================================')
            print('model saved to %s' % (save_point + saved_model_fn + '.t7'))

         ################### save model based on best top1 error
        if top3error_train  < best_top3_train :
            print('Saving model')
            best_top3_train = top3error_train
            best_model = copy.deepcopy(model)
            state = {
                'model': best_model,
                'top3' : best_top3_train,
                'args': args
            }
            if not os.path.isdir('checkpoint_train'):
                os.mkdir('checkpoint_top3_train')
            save_point = './checkpoint_top3_train/'
            if not os.path.isdir(save_point):
                os.mkdir(save_point)

            torch.save(state, save_point + saved_model_fn + '.t7')
            print('=======================================================================')
            print('model saved to %s' % (save_point + saved_model_fn + '.t7'))
