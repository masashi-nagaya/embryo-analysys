'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

from __future__ import print_function

import argparse
import os
import numpy as np
import random
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
parser.add_argument('--test-batch', default=1, type=int, metavar='N',
                    help='test batchsize (default: 1)')
# Checkpoints
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

parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--val_num', default=0, type=int,
                    help='val')

parser.add_argument('--iic', action='store_true',
                    help='non-iic or iic')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
random.seed(1)
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed_all(1)

class PU_IIC_Net(nn.Module):
    def __init__(self,model):
        super(PU_IIC_Net,self).__init__()
        self.PInet = model
        self.fc1 = nn.Linear(2048, 2)

    def forward(self,x):
        ax, rx, rx_m = self.PInet(x)

        x = self.fc1(rx_m)          

        return ax,rx,rx_m

class EGGDataset(Dataset):
    def __init__(self, csv_file,epoch, transform=None):
        self.frame = pd.read_csv(csv_file,header=None)
        self.transform = transform
        self.epoch = epoch

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):    

        dir   = self.frame.iloc[idx, 0].split()[0]
        movie_name = dir.split("/")[-2]        

        deg = random.randint(0,360)

        image1 = Image.open(self.frame.iloc[idx, 0].split()[0])
        image1 = image1.convert('L').convert('RGB')

        image1 = image1.rotate(deg)

        label = int(self.frame.iloc[idx, 0].split()[1]) 

        if self.transform:
            image1 = self.transform(image1)
                                    
        return image1,label,movie_name

def main():

    model = models.__dict__[args.arch]()

    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    model.module.att_conv2 = nn.Conv2d(1000, 2, kernel_size=1, padding=0,bias=False)
    model.module.fc = nn.Linear(2048, 2)
    print(args.iic)
    if args.iic == True:
        model = PU_IIC_Net(model)
    model = model.cuda() 
    
    # Resume
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume, map_location="cuda:"+str(args.gpu_id))
        model.load_state_dict(checkpoint['state_dict'])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    model.eval()

    if not "feature" in os.listdir("./"):
        os.mkdir("feature")
    
    if not "train"+str(args.val_num) in os.listdir("./feature"):
        os.mkdir("feature/train"+str(args.val_num))

    if not "test"+str(args.val_num) in os.listdir("./feature"):
        os.mkdir("feature/test"+str(args.val_num))

    #train_sawada_extract
    train_dataset = EGGDataset(csv_file="./csv/train"+str(args.val_num)+"_sawada.csv",epoch=1,transform=transforms.Compose([                           
                                transforms.ToTensor(),
                                normalize,
                                ]))
    
    train_loader = torch.utils.data.DataLoader(
                                train_dataset,batch_size=args.test_batch, shuffle=False,num_workers=args.workers, pin_memory=True)

    frame = pd.read_csv("./csv/train"+str(args.val_num)+"_sawada.csv")

    for i in range(4):  
        current_movie = frame.iloc[0, 0].split()[0].split("/")[-2]
        arr = []
        current_label = [1]
        for batch_idx, (inputs1, targets,movie_name)  in enumerate(train_loader):

            if batch_idx == 0:
                current_movie = movie_name

            if use_cuda:
                inputs1, targets = inputs1.cuda(),targets.cuda()
            inputs1, targets = torch.autograd.Variable(inputs1, volatile=True), torch.autograd.Variable(targets)

            if (batch_idx+i)%4==0:
                with torch.no_grad():
                    att_outputs1, outputs1, feature = model(inputs1)
                
                if current_movie != movie_name or len(frame)==batch_idx+1:
                    #print(current_movie)
                    #print("label:{}".format(targets.detach().cpu().clone().numpy()))
                    np.savez("./feature/train"+str(args.val_num)+"/"+current_movie[0]+"_"+str(i)+'.npz',arr,current_label)
                    arr = []
                
                arr.append(feature.detach().cpu().clone().numpy())
                current_label = targets.detach().cpu().clone().numpy()

                current_movie = movie_name

    #train_ncu_extract
    train_dataset = EGGDataset(csv_file="./csv/train"+str(args.val_num)+"_ncu.csv",epoch=1,transform=transforms.Compose([                           
                                transforms.ToTensor(),
                                normalize,
                                ]))
    
    train_loader = torch.utils.data.DataLoader(
                                train_dataset,batch_size=args.test_batch, shuffle=False,num_workers=args.workers, pin_memory=True)

    frame = pd.read_csv("./csv/train"+str(args.val_num)+"_ncu.csv")

    for i in range(6):  
        current_movie = frame.iloc[0, 0].split()[0].split("/")[-2]
        arr = []
        current_label = [1]
        for batch_idx, (inputs1, targets,movie_name)  in enumerate(train_loader):

            if batch_idx == 0:
                current_movie = movie_name

            if use_cuda:
                inputs1, targets = inputs1.cuda(),targets.cuda()
            inputs1, targets = torch.autograd.Variable(inputs1, volatile=True), torch.autograd.Variable(targets)

            if (batch_idx+i)%6==0:
                with torch.no_grad():
                    att_outputs1, outputs1, feature = model(inputs1)
                
                if current_movie != movie_name or len(frame)==batch_idx+1:
                    #print(current_movie)
                    #print("label:{}".format(targets.detach().cpu().clone().numpy()))
                    np.savez("./feature/train"+str(args.val_num)+"/"+current_movie[0]+"_"+str(i)+'.npz',arr,current_label)
                    arr = []
                
                arr.append(feature.detach().cpu().clone().numpy())
                current_label = targets.detach().cpu().clone().numpy()

                current_movie = movie_name

    #val_sawada_extract
    val_dataset = EGGDataset(csv_file="./csv/test"+str(args.val_num)+"_sawada.csv",epoch=1,transform=transforms.Compose([                           
                                transforms.ToTensor(),
                                normalize,
                                ]))
    
    val_loader = torch.utils.data.DataLoader(
                                val_dataset,batch_size=args.test_batch, shuffle=False,num_workers=args.workers, pin_memory=True)

    frame = pd.read_csv("./csv/test"+str(args.val_num)+"_sawada.csv")

    current_movie = frame.iloc[0, 0].split()[0].split("/")[-2]
    arr = []
    current_label = [1]
    for batch_idx, (inputs1, targets,movie_name)  in enumerate(val_loader):

        if batch_idx == 0:
            current_movie = movie_name

        if use_cuda:
            inputs1, targets = inputs1.cuda(),targets.cuda()
        inputs1, targets = torch.autograd.Variable(inputs1, volatile=True), torch.autograd.Variable(targets)

        if (batch_idx+i)%4==0:
            with torch.no_grad():
                att_outputs1, outputs1, feature = model(inputs1)
            
            if current_movie != movie_name or len(frame)==batch_idx+1:
                #print(current_movie)
                #print("label:{}".format(targets.detach().cpu().clone().numpy()))
                np.savez("./feature/test"+str(args.val_num)+"/"+current_movie[0]+"_"+str(i)+'.npz',arr,current_label)
                arr = []
            
            arr.append(feature.detach().cpu().clone().numpy())
            current_label = targets.detach().cpu().clone().numpy()

            current_movie = movie_name

    np.savez("./feature/test"+str(args.val_num)+"/"+current_movie[0]+'.npz',arr,current_label) 

    #val_ncu_extract
    val_dataset = EGGDataset(csv_file="./csv/test"+str(args.val_num)+"_ncu.csv",epoch=1,transform=transforms.Compose([                           
                                transforms.ToTensor(),
                                normalize,
                                ]))
    
    val_loader = torch.utils.data.DataLoader(
                                val_dataset,batch_size=args.test_batch, shuffle=False,num_workers=args.workers, pin_memory=True)

    frame = pd.read_csv("./csv/test"+str(args.val_num)+"_ncu.csv")
    current_movie = frame.iloc[0, 0].split()[0].split("/")[-2]
    arr = []
    current_label = [1]
    for batch_idx, (inputs1, targets,movie_name)  in enumerate(val_loader):

        if batch_idx == 0:
            current_movie = movie_name

        if use_cuda:
            inputs1, targets = inputs1.cuda(),targets.cuda()
        inputs1, targets = torch.autograd.Variable(inputs1, volatile=True), torch.autograd.Variable(targets)

        if (batch_idx+i)%6==0:
            with torch.no_grad():
                att_outputs1, outputs1, feature = model(inputs1)
            
            if current_movie != movie_name or len(frame)==batch_idx+1:
                #print(current_movie)
                #print("label:{}".format(targets.detach().cpu().clone().numpy()))
                np.savez("./feature/test"+str(args.val_num)+"/"+current_movie[0]+"_"+str(i)+'.npz',arr,current_label)
                arr = []
            
            arr.append(feature.detach().cpu().clone().numpy())
            current_label = targets.detach().cpu().clone().numpy()
            current_movie = movie_name

    np.savez("./feature/test"+str(args.val_num)+"/"+current_movie[0]+'.npz',arr,current_label)        
    
if __name__ == '__main__':
    main()
