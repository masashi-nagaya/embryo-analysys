'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import time
import numpy as np
import random
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.imagenet as customized_models
from torch.utils.data import Dataset
import pandas as pd
import torch.nn.functional as F

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
parser.add_argument('--mse_per_weight', default=1, type=float,
                    metavar='W', help='mse_weight(default: 10)')
parser.add_argument('--mse_att_weight', default=1, type=float,
                    metavar='W', help='mse_weight(default: 10)')             
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
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--beta', '-B', default=0.05, type=float,
                    help='Beta parameter of nnPU')
parser.add_argument('--gamma_pu', '-G', default=0.05, type=float,
                    help='Gamma parameter of nnPU')
parser.add_argument('--prior', '-pr', type=float, default=0.40,
                    help='prior')
parser.add_argument('--val_num', default=1, type=int,
                    help='val')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
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

            if self.transform:
                image1 = self.transform(image1)
                image2 = self.transform(image2)
    
            return image1,image2,label

        elif int(self.frame.iloc[idx, 0].split()[1]) == -5 : 
            image1 = Image.open(self.frame.iloc[idx-1, 0].split()[0])
            image1 = image1.convert('L').convert('RGB')

            image2 = Image.open(self.frame.iloc[idx-2, 0].split()[0])
            image2 =image2.convert('L').convert('RGB')

            label = int(self.frame.iloc[idx-1, 0].split()[1])

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

        if self.transform:
            image1 = self.transform(image1)

        return image1,label

class PULoss(nn.Module):

    def __init__(self, prior,gamma,beta):
        super(PULoss, self).__init__()
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, outputs, labels,num_positive , num_unlabeled):
        positive, unlabeled = labels == 1, labels == -1

        y_positive = self.sigmoid(-outputs)
        y_unlabeled = self.sigmoid(outputs)

        positive_risk = torch.sum(self.prior / num_positive * torch.dot(positive.float(),torch.squeeze(y_positive)))
        negative_risk = torch.sum(torch.dot(unlabeled.float(),torch.squeeze(y_unlabeled)) / num_unlabeled - self.prior / num_positive * torch.dot(positive.float(),torch.squeeze(y_unlabeled)))

        return positive_risk, negative_risk

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

    # create model
    model = models.__dict__[args.arch]()
    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    criterion_mse = nn.MSELoss().cuda()
    criterion_pu = PULoss(prior=args.prior,gamma=args.gamma,beta=args.beta).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Resume
    title = 'ImageNet-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        print(args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
    logger1 = Logger("pu-train"+str(args.val_num)+"_mse"+str(args.mse_att_weight)+'_seed'+str(args.manualSeed)+'.txt', title=title)
    logger1.set_names(['pu_loss_att','pu_loss_per','mse_loss','accuracy'])
    logger2 = Logger("pu-val"+str(args.val_num)+"_mse"+str(args.mse_att_weight)+'_seed'+str(args.manualSeed)+'.txt', title=title)
    logger2.set_names(['pu_loss_att','pu_loss_per','accuracy'])
 
    #model.module.att_conv = nn.Conv2d(512 * 4,chanel_num, kernel_size=1, padding=0,bias=False)
    model.module.att_conv2 = nn.Conv2d(1000, 2, kernel_size=1, padding=0,bias=False)
    model.module.fc = nn.Linear(2048, 2)
    model = model.cuda() 

    # Train and val
    for epoch in range(start_epoch, args.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        pu_losses_per, pu_losses_att, mse_losses, train_acc = train(train_loader, model ,criterion_mse,criterion_pu, optimizer, epoch, use_cuda)
        pu_losses_per_val, pu_losses_att_val , val_acc = val(val_loader, model ,criterion_pu,  use_cuda)

        logger1.append([pu_losses_att, pu_losses_per ,mse_losses, train_acc])
        logger2.append([pu_losses_att_val,pu_losses_per_val, val_acc])
        
        save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, checkpoint=args.checkpoint)
        
    logger1.close()
    logger1.plot()
    logger2.close()
    logger2.plot()
    
def train(train_loader, model, criterion_mse,criterion_pu, optimizer, epoch, use_cuda):

    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    per_losses = AverageMeter()
    att_losses = AverageMeter()
    mse_losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    
    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs1, inputs2, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs1, inputs2, targets = inputs1.cuda(),inputs2.cuda(), targets.cuda(async=True)
        inputs1,inputs2, targets = torch.autograd.Variable(inputs1),torch.autograd.Variable(inputs2), torch.autograd.Variable(targets)

        # compute output
        att_outputs1, outputs1, _  = model(inputs1)
        att_outputs2, outputs2, _  = model(inputs2)
        
        pos_num = 0
        un_num = 0
        for label in targets:
            if label == 1:
                pos_num += 1
            if label == -1:
                un_num += 1

        if pos_num == 0:
            pos_num = 1
        if un_num == 0:
            un_num = 1

        outputs1 = torch.chunk(outputs1,2,dim=1)
        outputs2 = torch.chunk(outputs2,2,dim=1)
        att_outputs1 = torch.chunk(att_outputs1,2,dim=1)
        att_outputs2 = torch.chunk(att_outputs2,2,dim=1)

        positive_risk_per1,negative_risk_per1 = criterion_pu(outputs1[1], targets,pos_num, un_num)
        positive_risk_att1,negative_risk_att1 = criterion_pu(att_outputs1[1], targets,pos_num, un_num)

        per_loss1 = positive_risk_per1 + negative_risk_per1
        att_loss1 = positive_risk_att1 + negative_risk_att1
    
        att_mse = criterion_mse(att_outputs2[1],att_outputs1[1].detach())
        per_mse = criterion_mse(outputs1[1],outputs2[1].detach())

        if negative_risk_att1 < -1*args.beta and negative_risk_per1 < -1*args.beta :
            loss = args.mse_att_weight * att_mse + args.mse_per_weight * per_mse - args.gamma_pu * negative_risk_att1 - args.gamma_pu * negative_risk_per1
            #print("both negative")

        elif negative_risk_att1 < -1*args.beta:
            loss = args.mse_att_weight * att_mse + args.mse_per_weight * per_mse - args.gamma_pu * negative_risk_att1 + per_loss1
            #print("attention negative")

        elif negative_risk_per1 < -1*args.beta:
            loss = args.mse_att_weight * att_mse + args.mse_per_weight * per_mse + att_loss1 - args.gamma_pu * negative_risk_per1
            #print("per negative")

        else:
            loss = args.mse_att_weight * att_mse + args.mse_per_weight * per_mse + per_loss1 + att_loss1
            #print("all positive")

        # compute gradient and do SGD step      
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # measure accuracy and record loss
        outputs1 = torch.sign(outputs1[1].data)
    
        all_count = 0
        acc_count = 0
        for i  in range(len(targets.detach().cpu().clone().numpy())):
            if targets.detach().cpu().clone().numpy()[i] == outputs1.detach().cpu().clone().numpy()[i]:
                acc_count += 1
            all_count += 1

        prec1 = acc_count/all_count*100

        per_losses.update(per_loss1, inputs1.size(0))
        att_losses.update(att_loss1, inputs1.size(0))
        mse_losses.update(per_mse, inputs1.size(0))
        top1.update(prec1, inputs1.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) |Total: {total:} | ETA: {eta:} | per_Loss: {per_loss:.4f} | att_Loss: {att_loss:.4f} | mse_Loss: {mse_loss:.4f} | top1: {top1: .4f} '.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    per_loss=per_losses.avg,
                    att_loss=att_losses.avg,
                    mse_loss=mse_losses.avg,
                    top1=top1.avg
                    )
        bar.next()
    bar.finish()
    return (per_losses.avg, att_losses.avg , mse_losses.avg, top1.avg)

def val(val_loader, model,criterion_pu, use_cuda):

    # switch to train mode
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    per_losses = AverageMeter()
    att_losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs1,  targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs1,  targets = inputs1.cuda(), targets.cuda(async=True)
        inputs1,targets = torch.autograd.Variable(inputs1), torch.autograd.Variable(targets)

        # compute output
        with torch.no_grad():
            att_outputs1, outputs1, _  = model(inputs1)
        
        pos_num = 0
        un_num = 0
        for label in targets:
            if label == 1:
                pos_num += 1
            if label == -1:
                un_num += 1

        if pos_num == 0:
            pos_num = 1
        if un_num == 0:
            un_num = 1

        outputs1 = torch.chunk(outputs1,2,dim=1)
        att_outputs1 = torch.chunk(att_outputs1,2,dim=1)

        positive_risk_per1,negative_risk_per1 = criterion_pu(outputs1[1], targets,pos_num, un_num)
        positive_risk_att1,negative_risk_att1 = criterion_pu(att_outputs1[1], targets,pos_num, un_num)

        per_loss1 = positive_risk_per1 + negative_risk_per1
        att_loss1 = positive_risk_att1 + negative_risk_att1
    
        # measure accuracy and record loss
        outputs1 = torch.sign(outputs1[1].data)
    
        all_count = 0
        acc_count = 0
        for i  in range(len(targets.detach().cpu().clone().numpy())):
            if targets.detach().cpu().clone().numpy()[i] == outputs1.detach().cpu().clone().numpy()[i]:
                acc_count += 1
            all_count += 1

        prec1 = acc_count/all_count*100

        per_losses.update(per_loss1, inputs1.size(0))
        att_losses.update(att_loss1, inputs1.size(0))
        top1.update(prec1, inputs1.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) |Total: {total:} | ETA: {eta:} | per_Loss: {per_loss:.4f} | att_Loss: {att_loss:.4f} | top1: {top1: .4f} '.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    per_loss=per_losses.avg,
                    att_loss=att_losses.avg,
                    top1=top1.avg
                    )
        bar.next()
    bar.finish()
    return (per_losses.avg , att_losses.avg, top1.avg)

def save_checkpoint(state , checkpoint='checkpoint',  filename="pu"+str(args.val_num)+"_mse"+str(args.mse_per_weight)+'_seed'+str(args.manualSeed)+'.pth.tar'):
    filepath = os.path.join(checkpoint, filename)   
    torch.save(state, filepath)

if __name__ == '__main__':
    main()


