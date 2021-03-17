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
from matplotlib import pylab as plt

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
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--test-batch', default=1, type=int, metavar='N',
                    help='test batchsize (default: 1)')
# Checkpoints
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

parser.add_argument('--test_mode', default='pn', type=str,
                    choices=["pn","nnpu","auc-pr","auc-pr-nnpu"],
                    help='pn or nnpu or auc-pr or auc-pr-nnpu')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()


if use_cuda:
    torch.cuda.manual_seed_all(1)

class EGGDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
            
        dir   = self.frame.iloc[idx, 0].split()[0]
        movie_name = dir.split("/")[-2]
        image_name = dir.split("/")[-1]

        image1 = Image.open(self.frame.iloc[idx, 0].split()[0])
        image = image1.convert('L').convert('RGB')

        label = int(self.frame.iloc[idx, 0].split()[1])

        if label == -1:
            label = 0

        if self.transform:
            image1 = self.transform(image)

        image = np.array(image)
                                
        return image,image1,label,movie_name,image_name 

def main():
    
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # define loss function (criterion) and optimizer
    model.module.att_conv2 = nn.Conv2d(1000, 2, kernel_size=1, padding=0,bias=False)
    model.module.fc = nn.Linear(2048, 2)
    model = model.cuda() 

    if not "attention"+str(args.val_num) in os.listdir("./"):
        os.mkdir("attention"+str(args.val_num))
        os.mkdir("attention"+str(args.val_num)+"/birth")
        os.mkdir("attention"+str(args.val_num)+"/failure")

    # Resume
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        print(args.resume)
        checkpoint = torch.load(args.resume,map_location="cuda:"+str(args.gpu_id))
        model.load_state_dict(checkpoint['state_dict'])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    val_dataset = EGGDataset(csv_file="./csv/test"+str(args.val_num)+".csv",transform=transforms.Compose([                         
                                transforms.ToTensor(),
                                normalize,
                                ]))  

    val_loader = torch.utils.data.DataLoader(
                                val_dataset,batch_size=args.test_batch, shuffle=False,num_workers=args.workers, pin_memory=True)
    
    model.eval()
    sigmoid = nn.Sigmoid()
    for batch_idx, (image,inputs1,targets,movie_name,image_name)  in enumerate(val_loader):

        if use_cuda:
            inputs1, targets = inputs1.cuda(),targets.cuda()
        inputs1, targets = torch.autograd.Variable(inputs1, volatile=True), torch.autograd.Variable(targets)

        label =  targets.detach().cpu().clone().numpy()[0]
        if label == 1:
            if not movie_name[0] in os.listdir("./attention"+str(args.val_num)+"/birth/"):
                os.mkdir("./attention"+str(args.val_num)+"/birth/"+movie_name[0])

        if label == 0:
            if not movie_name[0] in os.listdir("./attention"+str(args.val_num)+"/failure/"):
                os.mkdir("./attention"+str(args.val_num)+"/failure/"+movie_name[0])

        att_outputs1, per_outputs1, attention = model(inputs1)

        if args.test_mode == "pn":
            pos_prob1 = sigmoid(per_outputs1).detach().cpu().clone().numpy()[0][1]
        else:
            pos_prob1 = sigmoid(per_outputs1).detach().cpu().clone().numpy()[0][1]

        attention, fe, per = attention
        
        c_att = attention.data.cpu()
        c_att = c_att.numpy()
        d_inputs = image.data.cpu()
        d_inputs = d_inputs.numpy()

        for item_img, item_att in zip(d_inputs, c_att):

            v_img = item_img[:, :, ::-1]

            resize_att = cv2.resize(item_att[0], (224, 224))
            resize_att = np.maximum(resize_att, 0)
            resize_att = resize_att / np.max(resize_att)

            resize_att *= 255.

            cv2.imwrite('stock1.png', v_img)
            cv2.imwrite('stock2.png', resize_att)
            v_img = cv2.imread('stock1.png')
            vis_map = cv2.imread('stock2.png', 0)
            jet_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
            #jet_map = cv2.add(v_img, jet_map)
            jet_map = np.float32(v_img) + np.float32(jet_map)

            jet_map = 255 * jet_map / np.max(jet_map)

            if label == 1:
                cv2.imwrite("./attention"+str(args.val_num)+"/birth/"+movie_name[0]+"/"+image_name[0].split(".")[0]+"_"+str(pos_prob1)+".jpg", jet_map)

            if label == 0:
                cv2.imwrite("./attention"+str(args.val_num)+"/failure/"+movie_name[0]+"/"+image_name[0].split(".")[0]+"_"+str(pos_prob1)+".jpg", jet_map)
    
if __name__ == '__main__':
    main()
