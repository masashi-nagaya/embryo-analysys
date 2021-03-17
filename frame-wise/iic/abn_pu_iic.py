'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import numpy as np
import random
import cv2
from PIL import Image
import sys

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
parser.add_argument('--train-batch', default=8, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=8, type=int, metavar='N',
                    help='test batchsize (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate')
             
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
parser.add_argument('--manualSeed',default=1, type=int, help='manual seed')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--val_num', default=1, type=int,
                    help='val')

parser.add_argument('--beta', '-B', default=0.05, type=float,
                    help='Beta parameter of nnPU')
parser.add_argument('--gamma_pu', '-G', default=0.05, type=float,
                    help='Gamma parameter of nnPU')
parser.add_argument('--prior', '-pr', type=float, default=0.3,
                    help='prior')   

parser.add_argument('--mse_per_weight', '--per_w', default=1, type=float,
                    metavar='W', help='mse_weight(default: 10)')
parser.add_argument('--mse_att_weight', '--att_w', default=1, type=float,
                    metavar='W', help='mse_weight(default: 10)')        
parser.add_argument('--iic_weight', '-iic_w', default=1, type=float)

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

class PU_IIC_Net(nn.Module):
    def __init__(self,model):
        super(PU_IIC_Net,self).__init__()
        self.PInet = model
        self.fc1 = nn.Linear(2048, 2)

    def forward(self,x):
        ax, rx, rx_m,att = self.PInet(x)

        x = self.fc1(rx_m)          

        return ax,rx,x

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
        objective = positive_risk + negative_risk

        return positive_risk, negative_risk

def IID_loss(x_out, x_tf_out, lamb=1, EPS=sys.float_info.epsilon):
# has had softmax applied
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k,k)  # but should be same, symmetric

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    """
    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS
    """
    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device = p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device = p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device = p_i.device), p_i)
    
    loss = -p_i_j * (torch.log(p_i_j) \
                    - lamb * torch.log(p_j))

    loss = loss.sum()

    return loss

def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)

    bn, k = x_out.size()

    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j

class EGGDataset(Dataset):
    def __init__(self, csv_file, transform=None,transform_aug=None,train="train"):
        self.frame = pd.read_csv(csv_file,header=None)
        self.transform = transform
        self.transform_aug = transform_aug
        self.train=train

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if self.train == "train":
            if int(self.frame.iloc[idx, 0].split()[1]) != -5 and int(self.frame.iloc[idx+1, 0].split()[1]) != -5:

                image1 = Image.open(self.frame.iloc[idx, 0].split()[0])
                image1 = image1.convert('L').convert('RGB')

                image2 = Image.open(self.frame.iloc[idx+1, 0].split()[0])         
                image2 =image2.convert('L').convert('RGB')

                label = int(self.frame.iloc[idx, 0].split()[1])

                image_t1 = self.transform(image1)
                image_aug_t1 = self.transform_aug(image1)
                image_aug_t2 =  self.transform_aug(image2)

                return image_t1,image_aug_t1,image_aug_t2,label

            elif int(self.frame.iloc[idx, 0].split()[1]) == -5 : 
                image1 = Image.open(self.frame.iloc[idx-1, 0].split()[0])
                image1 = image1.convert('L').convert('RGB')

                image2 = Image.open(self.frame.iloc[idx-2, 0].split()[0])
                image2 =image2.convert('L').convert('RGB')

                label = int(self.frame.iloc[idx-1, 0].split()[1])

                image_t1 = self.transform(image1)
                image_aug_t1 = self.transform_aug(image1)
                image_aug_t2 =  self.transform_aug(image2)

                return image_t1,image_aug_t1,image_aug_t2,label

            elif int(self.frame.iloc[idx+1, 0].split()[1]) == -5:
                image1 = Image.open(self.frame.iloc[idx, 0].split()[0])
                image1 = image1.convert('L').convert('RGB')

                image2 = Image.open(self.frame.iloc[idx-1, 0].split()[0])
                image2 = image2.convert('L').convert('RGB')

                label = int(self.frame.iloc[idx-1, 0].split()[1])            
            
                image_t1 = self.transform(image1)
                image_aug_t1 = self.transform_aug(image1)
                image_aug_t2 =  self.transform_aug(image2)

                return image_t1,image_aug_t1,image_aug_t2,label

        else:
            if idx +1 != len(self.frame):
                image1 = Image.open(self.frame.iloc[idx, 0].split()[0])
                image1 = image1.convert('L').convert('RGB')

                image2 = Image.open(self.frame.iloc[idx+1, 0].split()[0])
                image2 = image2.convert('L').convert('RGB')

                label = int(self.frame.iloc[idx, 0].split()[1])

                image_t1 = self.transform(image1)
                image_aug_t1 = self.transform_aug(image1)
                image_aug_t2 =  self.transform_aug(image2)

                return image_t1,image_aug_t1,image_aug_t2,label

            else:
                image1 = Image.open(self.frame.iloc[idx, 0].split()[0])
                image1 = image1.convert('L').convert('RGB')

                image2 = Image.open(self.frame.iloc[idx-1, 0].split()[0])
                image2 = image2.convert('L').convert('RGB')

                label = int(self.frame.iloc[idx, 0].split()[1])

                image_t1 = self.transform(image1)
                image_aug_t1 = self.transform_aug(image1)
                image_aug_t2 =  self.transform_aug(image2)

                return image_t1,image_aug_t1,image_aug_t2,label


def main():
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = EGGDataset(csv_file="./csv/train"+str(args.val_num)+".csv",
                                train="train",
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                normalize
                                ]),
                                transform_aug=transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(360),
                                transforms.ToTensor(),
                                normalize
                                ]))

    train_loader = torch.utils.data.DataLoader(
                                train_dataset,batch_size=args.train_batch, shuffle=True,
                                num_workers=args.workers, pin_memory=True)

    val_dataset = EGGDataset(csv_file="./csv/test"+str(args.val_num)+".csv",
                                train="val",
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                normalize
                                ]),
                                transform_aug=transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(360),
                                transforms.ToTensor(),
                                normalize
                                ]))

    val_loader = torch.utils.data.DataLoader(
                                val_dataset,batch_size=args.test_batch, shuffle=True,
                                num_workers=args.workers, pin_memory=True)       

    # create model
    model = models.__dict__[args.arch]()
    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion_mse = nn.MSELoss().cuda()
    criterion_pu = PULoss(prior=args.prior,gamma=args.gamma_pu,beta=args.beta).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Resume
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        print(args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
    logger1 = Logger('pu_iic-train'+str(args.val_num)+"_mse"+str(args.mse_per_weight)+'_iic'+str(args.iic_weight)+'_seed'+str(args.manualSeed)+'.txt')
    logger1.set_names(['att_loss','per_loss','iic_loss','mse_loss','accuracy'])
    logger2 = Logger('pu_iic-val'+str(args.val_num)+"_mse"+str(args.mse_per_weight)+'_iic'+str(args.iic_weight)+'_seed'+str(args.manualSeed)+'.txt')
    logger2.set_names(['att_loss','per_loss','iic_loss','mse_loss','accuracy'])

    model.module.att_conv2 = nn.Conv2d(1000, 2, kernel_size=1, padding=0,bias=False)
    model.module.fc = nn.Linear(2048, 2)
    model = PU_IIC_Net(model)
    model = model.cuda()  

    # Train and val
    for epoch in range(start_epoch, args.epochs):

        per_losses, att_losses , iic_losses, mse_losses, acc = train(train_loader, model ,criterion_mse,criterion_pu, optimizer, epoch, use_cuda)
        per_losses_val, att_losses_val , iic_losses_val, mse_losses_val, acc_val = val(val_loader, model ,criterion_mse,criterion_pu, use_cuda)

        logger1.append([per_losses, att_losses ,iic_losses, mse_losses, acc])
        logger2.append([per_losses_val, att_losses_val ,iic_losses_val, mse_losses_val, acc])      

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
    iic_losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    softmax = nn.Softmax()
    
    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (image_t1,image_aug_t1,image_aug_t2, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            image_t1, image_aug_t1,image_aug_t2, targets = image_t1.cuda(),image_aug_t1.cuda(),image_aug_t2.cuda(), targets.cuda(async=True)
        image_t1, image_aug_t1,image_aug_t2, targets = torch.autograd.Variable(image_t1),\
                                                        torch.autograd.Variable(image_aug_t1),\
                                                        torch.autograd.Variable(image_aug_t2), \
                                                        torch.autograd.Variable(targets)

        # compute output
        att1, per1, iic1  = model(image_t1)
        att_aug1, per_aug1, iic_aug1  = model(image_aug_t1)
        att_aug2, per_aug2, iic_aug2  = model(image_aug_t2)
        
        iic_prob1 = softmax(iic1)
        iic_aug_prob1 = softmax(iic_aug1)   

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

        per_aug1 = torch.chunk(per_aug1,2,dim=1)
        per_aug2 = torch.chunk(per_aug2,2,dim=1)
        att_aug1 = torch.chunk(att_aug1,2,dim=1)
        att_aug2 = torch.chunk(att_aug2,2,dim=1)

        p_risk_per1,n_risk_per1 = criterion_pu(per_aug1[1], targets,pos_num, un_num)
        p_risk_att1,n_risk_att1 = criterion_pu(att_aug1[1], targets,pos_num, un_num)

        per_loss1 = p_risk_per1 + n_risk_per1
        att_loss1 = p_risk_att1 + n_risk_att1

        iic_loss1 = IID_loss(iic_prob1, iic_aug_prob1)

        att_mse = criterion_mse(att_aug1[1],att_aug2[1].detach())
        per_mse = criterion_mse(per_aug1[1],per_aug2[1].detach())

        if n_risk_att1 < -1*args.beta and n_risk_per1 < -1*args.beta :
            loss = - args.gamma_pu * n_risk_att1 - args.gamma_pu * n_risk_per1 \
                   + args.mse_att_weight * att_mse + args.mse_per_weight * per_mse \
                   + args.iic_weight*iic_loss1

        elif n_risk_att1 < -1*args.beta:
            loss = - args.gamma_pu * n_risk_att1 + per_loss1 \
                   + args.mse_att_weight * att_mse + args.mse_per_weight * per_mse \
                   + args.iic_weight*iic_loss1
                    
        elif n_risk_per1 < -1*args.beta:
            loss = att_loss1 - args.gamma_pu * n_risk_per1 \
                   + args.mse_att_weight * att_mse + args.mse_per_weight * per_mse \
                   + args.iic_weight*iic_loss1

        else:
            loss =  per_loss1 + att_loss1 \
                    + args.mse_att_weight * att_mse + args.mse_per_weight * per_mse \
                    + args.iic_weight*iic_loss1

        # compute gradient and do SGD step      
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # measure accuracy and record loss
        per_aug1 = torch.sign(per_aug1[1].data)
    
        all_count = 0
        acc_count = 0
        
        for i  in range(len(targets.detach().cpu().clone().numpy())):
            if int(targets.detach().cpu().clone().numpy()[i]) == int(per_aug1.detach().cpu().clone().numpy()[i][0]):
                acc_count += 1
            all_count += 1

        prec1 = float(acc_count)/all_count*100

        per_losses.update(per_loss1, image_t1.size(0))
        att_losses.update(att_loss1, image_t1.size(0))
        mse_losses.update(per_mse, image_t1.size(0))
        iic_losses.update(iic_loss1, image_t1.size(0))
        top1.update(prec1, image_t1.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) |Total: {total:} | ETA: {eta:} | per_Loss: {per_loss:.4f} | att_Loss: {att_loss:.4f} | mse_Loss: {mse_loss:.4f} | iic_Loss: {iic_loss:.4f} | top1: {top1: .4f} '.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    per_loss=per_losses.avg,
                    att_loss=att_losses.avg,
                    mse_loss=mse_losses.avg,
                    iic_loss=iic_losses.avg,
                    top1=top1.avg
                    )
        bar.next()
    bar.finish()
    return (per_losses.avg, att_losses.avg ,iic_losses.avg, mse_losses.avg, top1.avg)

def val(val_loader, model, criterion_mse,criterion_pu, use_cuda):

    # switch to train mode
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    per_losses = AverageMeter()
    att_losses = AverageMeter()
    mse_losses = AverageMeter()
    iic_losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    softmax = nn.Softmax()
    
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (image_t1,image_aug_t1,image_aug_t2, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            image_t1, image_aug_t1,image_aug_t2, targets = image_t1.cuda(),image_aug_t1.cuda(),image_aug_t2.cuda(), targets.cuda(async=True)
        image_t1, image_aug_t1,image_aug_t2, targets = torch.autograd.Variable(image_t1),\
                                                        torch.autograd.Variable(image_aug_t1),\
                                                        torch.autograd.Variable(image_aug_t2), \
                                                        torch.autograd.Variable(targets)

        # compute output
        with torch.no_grad():
            att1, per1, iic1  = model(image_t1)
            att_aug1, per_aug1, iic_aug1  = model(image_aug_t1)
            att_aug2, per_aug2, iic_aug2  = model(image_aug_t2)
        
        iic_prob1 = softmax(iic1)
        iic_aug_prob1 = softmax(iic_aug1)   

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

        per_aug1 = torch.chunk(per_aug1,2,dim=1)
        per_aug2 = torch.chunk(per_aug2,2,dim=1)
        att_aug1 = torch.chunk(att_aug1,2,dim=1)
        att_aug2 = torch.chunk(att_aug2,2,dim=1)

        p_risk_per1,n_risk_per1 = criterion_pu(per_aug1[1], targets,pos_num, un_num)
        p_risk_att1,n_risk_att1 = criterion_pu(att_aug1[1], targets,pos_num, un_num)

        per_loss1 = p_risk_per1 + n_risk_per1
        att_loss1 = p_risk_att1 + n_risk_att1

        iic_loss1 = IID_loss(iic_prob1, iic_aug_prob1)

        att_mse = criterion_mse(att_aug1[1],att_aug2[1].detach())
        per_mse = criterion_mse(per_aug1[1],per_aug2[1].detach())

        # measure accuracy and record loss
        per_aug1 = torch.sign(per_aug1[1].data)
    
        all_count = 0
        acc_count = 0
        
        for i  in range(len(targets.detach().cpu().clone().numpy())):
            if int(targets.detach().cpu().clone().numpy()[i]) == int(per_aug1.detach().cpu().clone().numpy()[i][0]):
                acc_count += 1
            all_count += 1

        prec1 = float(acc_count)/all_count*100

        per_losses.update(per_loss1, image_t1.size(0))
        att_losses.update(att_loss1, image_t1.size(0))
        mse_losses.update(per_mse, image_t1.size(0))
        iic_losses.update(iic_loss1, image_t1.size(0))
        top1.update(prec1, image_t1.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) |Total: {total:} | ETA: {eta:} | per_Loss: {per_loss:.4f} | att_Loss: {att_loss:.4f} | mse_Loss: {mse_loss:.4f} | iic_Loss: {iic_loss:.4f} | top1: {top1: .4f} '.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    per_loss=per_losses.avg,
                    att_loss=att_losses.avg,
                    mse_loss=mse_losses.avg,
                    iic_loss=iic_losses.avg,
                    top1=top1.avg
                    )
        bar.next()
    bar.finish()
    return (per_losses.avg, att_losses.avg ,iic_losses.avg, mse_losses.avg, top1.avg)

def save_checkpoint(state , checkpoint='checkpoint',  filename='pu'+str(args.val_num)+"_mse"+str(args.mse_per_weight)+'_iic'+str(args.iic_weight)+'_beta'+str(args.beta)+'_seed'+str(args.manualSeed)+'_ver3.pth.tar'):
    filepath = os.path.join(checkpoint, filename)   
    torch.save(state, filepath)

if __name__ == '__main__':
    main()


