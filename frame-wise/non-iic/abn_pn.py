'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import configparser
import models.imagenet as customized_models
import pandas as pd
import torch.nn.functional as F

from torch.utils.data import Dataset
from PIL import Image
from os import path

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))
for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

# model_names = default_model_names + customized_models_names
model_names = customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=16, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=8, type=int, metavar='N',
                    help='test batchsize (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--per_loss1', default=1, type=int,
                    metavar='W', help='att_loss1_weight(default: 1)')
parser.add_argument('--att_loss1', default=1, type=int,
                    metavar='W', help='att_loss1_weight(default: 1)')
parser.add_argument('--mse_loss_per', '--per_mse', default=1, type=float,
                    metavar='W', help='mse_per_weight(default: 1)')
parser.add_argument('--mse_loss_att', '--att_mse', default=1, type=float,
                    metavar='W', help='mse_att_weight(default: 1)')                  

# Checkpoints-a resnet50 
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
#Device options
parser.add_argument('--gpu-id', default='0', type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--val_num', default=0, type=int,
                    help='val')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

class EGGDataset_train(Dataset):
    def __init__(self, csv_file, transform=None):
        self.frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if int(self.frame.iloc[idx, 0].split()[1]) != -5 and int(self.frame.iloc[idx+1, 0].split()[1]) != -5:

            image1 = Image.open(self.frame.iloc[idx, 0].split()[0])
            image1 = image1.convert('L').convert('RGB')
            image2 = Image.open(self.frame.iloc[idx+1, 0].split()[0])
            
            image2 =image2.convert('L').convert('RGB')

            label = int(self.frame.iloc[idx, 0].split()[1])
            
            if label == -1:
                label = 0

            if self.transform:
                image1 = self.transform(image1)
                image2 = self.transform(image2)
       
            return image1,image2,label

        elif int(self.frame.iloc[idx, 0].split()[1]) == -5:
            image1 = Image.open(self.frame.iloc[idx-1, 0].split()[0])
            image1 = image1.convert('L').convert('RGB')

            image2 = Image.open(self.frame.iloc[idx-2, 0].split()[0])
            image2 =image2.convert('L').convert('RGB')

            label = int(self.frame.iloc[idx-1, 0].split()[1])

            if label == -1:
                label = 0

            if self.transform:
                image1 = self.transform(image1)
                image2 = self.transform(image2)       

            return image1,image2,label

        elif int(self.frame.iloc[idx+1, 0].split()[1]) == -5:
            image1 = Image.open(self.frame.iloc[idx, 0].split()[0])
            image1 = image1.convert('L').convert('RGB')

            image2 = Image.open(self.frame.iloc[idx-1, 0].split()[0])
            image2 = image2.convert('L').convert('RGB')

            label = int(self.frame.iloc[idx-1, 0].split()[1])

            if label == -1:
                label = 0
                
            if self.transform:
                image1 = self.transform(image1)
                image2 = self.transform(image2)
                
            return image1,image2,label

class EGGDataset_val(Dataset):
    def __init__(self, csv_file, transform=None):
        self.frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):

        image1 = Image.open(self.frame.iloc[idx, 0].split()[0])
        image1 = image1.convert('L').convert('RGB')

        label = int(self.frame.iloc[idx, 0].split()[1])
        
        if label == -1:
            label = 0

        if self.transform:
            image1 = self.transform(image1)
    
        return image1,label

def main():
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_dataset = EGGDataset_train(csv_file="./csv/train"+str(args.val_num)+".csv",transform=transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(360, resample=False, expand=False, center=None),
                                transforms.ToTensor(),
                                normalize,
                                ]))

    train_loader = torch.utils.data.DataLoader(
                                train_dataset,batch_size=args.train_batch, shuffle=True,
                                num_workers=args.workers, pin_memory=True)
    
    val_dataset = EGGDataset_val(csv_file="./csv/test"+str(args.val_num)+".csv",transform=transforms.Compose([
                                transforms.ToTensor(),
                                normalize,
                                ]))

    val_loader = torch.utils.data.DataLoader(
                                val_dataset,batch_size=args.test_batch, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
    
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    birth_count = 0
    failure_count = 0
    df = pd.read_csv("./csv/train"+str(args.val_num)+".csv",header=None)
    for index ,row in df.iterrows():
        output = row.str.split(' ')
        output = output.tolist()
        output = output[0][1]
        if output == "1":
            birth_count += 1
        elif output == "-1":
            failure_count += 1

    print(failure_count)
    print(birth_count)

    w = failure_count/birth_count

    weights = torch.tensor([1,w])
    criterion_ce = nn.CrossEntropyLoss(weight = weights).cuda()
    criterion_mse = nn.MSELoss().cuda() 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Resume
    title = 'ImageNet-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        print(args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        logger1 = Logger('pn-train'+str(args.val_num)+"_mse"+str(args.mse_loss_per)+"_seed"+str(args.manualSeed)+".txt", title=title)
        logger1.set_names(['ce_loss_att','ce_loss_per','mse_loss','top1' ])
        logger2 = Logger('pn-val'+str(args.val_num)+"_mse"+str(args.mse_loss_per)+"_seed"+str(args.manualSeed)+".txt", title=title)
        logger2.set_names(['ce_loss_att','ce_loss_per','top1' ])

    model.module.att_conv2 = nn.Conv2d(1000, 2, kernel_size=1, padding=0,bias=False)
    model.module.fc = nn.Linear(2048, 2)
    model = model.cuda()

    # Train and val
    for epoch in range(start_epoch, args.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        att_loss,per_loss,mse_loss,top1 = train(train_loader, model,criterion_ce ,criterion_mse, optimizer, epoch, use_cuda)
        att_loss_val,per_loss_val,top1_val = val(val_loader, model,criterion_ce,epoch,use_cuda)
        
        # append logger file
        logger1.append([att_loss,per_loss,mse_loss,top1])
        logger2.append([att_loss_val,per_loss_val,top1_val])

        save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                },checkpoint=args.checkpoint)

    logger1.close()
    logger1.plot()
    logger2.close()
    logger2.plot()
    
def train(train_loader, model,criterion_ce, criterion_mse, optimizer, epoch, use_cuda):

    model.train()

    softmax = nn.Softmax()
    att_losses = AverageMeter()
    per_losses = AverageMeter()
    mse_losses = AverageMeter()
    top1 = AverageMeter()
    
    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs1, inputs2, targets) in enumerate(train_loader):

        if use_cuda:
            inputs1, inputs2, targets = inputs1.cuda(),inputs2.cuda(), targets.cuda()
        inputs1, inputs2, targets = torch.autograd.Variable(inputs1),torch.autograd.Variable(inputs2), torch.autograd.Variable(targets)

        # compute output
        att_outputs1, outputs1, _  = model(inputs1)
        att_outputs2, outputs2, _  = model(inputs2)  

        mse_loss_att = criterion_mse(att_outputs2,att_outputs1.detach())
        mse_loss_per = criterion_mse(outputs1,outputs2.detach())

        att_loss1 = criterion_ce(att_outputs1, targets)
        per_loss1 = criterion_ce(outputs1, targets)

        loss =  args.att_loss1*att_loss1 + args.per_loss1*per_loss1 +  args.mse_loss_att*mse_loss_att + args.mse_loss_per*mse_loss_per 

        pos_prob = softmax(outputs1).detach().cpu().clone().numpy()
        
        correct_count = 0
        count = 0
        for i in range(len(pos_prob)):
            if targets[i] == 1:
                if pos_prob[i][1] >0.5:
                    correct_count += 1

            if targets[i] == 0:
                if pos_prob[i][1] <0.5:
                    correct_count += 1
            count += 1

        # measure accuracy and record loss
        top1.update(correct_count/count, inputs1.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        att_losses.update(att_loss1.data, inputs1.size(0))
        per_losses.update(per_loss1.data, inputs1.size(0))
        mse_losses.update(mse_loss_per.data, inputs1.size(0))

        top1.update(correct_count/count, inputs1.size(0))

        # plot progress
        bar.suffix  = '({batch}/{size})|att_loss: {att_loss:.4f}|ce_loss: {ce_loss:.4f}|mse_loss: {mse_loss:.4f} |top1: {top1: .4f} | '.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    att_loss=att_losses.avg,
                    ce_loss=per_losses.avg,
                    mse_loss=mse_losses.avg,
                    top1=top1.avg,
                    )
        bar.next()
    bar.finish()
    return ( att_losses.avg,per_losses.avg,mse_losses.avg,top1.avg)

def val(test_loader, model,criterion_ce,epoch, use_cuda):

    model.eval()
    softmax = nn.Softmax()
    per_losses = AverageMeter()
    att_losses = AverageMeter()
    top1 = AverageMeter()
    
    bar = Bar('Processing', max=len(test_loader))
    for batch_idx, (inputs1,targets) in enumerate(test_loader):

        if use_cuda:
            inputs1, targets = inputs1.cuda(), targets.cuda()

        # compute output
        with torch.no_grad():
            att_outputs1, outputs1, _  = model(inputs1) 

        per_loss1 = criterion_ce(outputs1, targets)
        att_loss1 = criterion_ce(att_outputs1, targets)

        pos_prob = softmax(outputs1).detach().cpu().clone().numpy()
        
        correct_count = 0
        count = 0
        for i in range(len(pos_prob)):
            if targets[i] == 1:
                if pos_prob[i][1] >0.5:
                    correct_count += 1

            if targets[i] == 0:
                if pos_prob[i][1] <0.5:
                    correct_count += 1
            count += 1

        # measure accuracy and record loss
        top1.update(correct_count/count, inputs1.size(0))
        per_losses.update(per_loss1.data, inputs1.size(0))
        att_losses.update(att_loss1.data, inputs1.size(0))

        # plot progress
        bar.suffix  = '({batch}/{size})|per_loss: {ce_loss:.4f}|top1: {top1: .4f} | '.format(
                    batch=batch_idx + 1,
                    size=len(test_loader),
                    ce_loss=per_losses.avg,
                    top1=top1.avg,
                    )
        bar.next()
    bar.finish()
    return (att_losses.avg,per_losses.avg,top1.avg)

def save_checkpoint(state,checkpoint='checkpoint', filename='pn'+str(args.val_num)+"_mse"+str(args.mse_loss_per)+"_seed"+str(args.manualSeed)+'.pth.tar'):
    filepath = os.path.join(checkpoint, filename)   
    torch.save(state, filepath)


if __name__ == '__main__':
    main()
