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
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.imagenet as customized_models
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score

from torch.utils.data import Dataset
import pandas as pd
import csv

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

parser.add_argument('--val_num', default=1, type=int,
                    help='val')

parser.add_argument('--seed_number', default=5, type=int,
                    help='seed Number of times')

parser.add_argument('--mode', default='pn', type=str,
                    choices=["pn","nnpu","auc-pr","auc-pr-nnpu"],
                    help='pn or nnpu or auc-pr or auc-pr-nnpu')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

class EGGDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.frame = pd.read_csv(csv_file,header=None)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):            
        dir   = self.frame.iloc[idx, 0].split()[0]
        movie_name = dir.split("/")[-2]

        image1 = Image.open(self.frame.iloc[idx, 0].split()[0])
        image1 = image1.convert('L').convert('RGB')

        label = int(self.frame.iloc[idx, 0].split()[1])

        if self.transform:
            image1 = self.transform(image1)
                                    
        return image1,label,movie_name

def main():
    model = models.__dict__[args.arch]()

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    model.module.att_conv2 = nn.Conv2d(1000, 2, kernel_size=1, padding=0,bias=False)
    model.module.fc = nn.Linear(2048, 2)
    model = model.cuda() 

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_dataset = EGGDataset(csv_file="./csv/test"+str(args.val_num)+".csv"  ,transform=transforms.Compose([                           
                                transforms.ToTensor(),
                                normalize,
                                ]))

    val_loader = torch.utils.data.DataLoader(
                                val_dataset,batch_size=args.test_batch, shuffle=False,num_workers=args.workers, pin_memory=True)

    frame = pd.read_csv("./csv/test"+str(args.val_num)+".csv",header=None)
    
    model.eval()
    thresholds = [i/100 for i in range(1,100,1)]
    a_all={}
    p_all = {}
    r_all = {}
    f_all = {}

    a_average_csv = []
    p_average_csv = []
    r_average_csv = []
    f_average_csv = []    
    pr_average_csv = ["auc-pr"]
    roc_average_csv = ["auc-roc"]
    for i in range(args.seed_number):
        checkpoint1 = torch.load(args.resume+str(i+1)+".pth.tar",map_location="cuda:"+str(args.gpu_id))
        model.load_state_dict(checkpoint1['state_dict'])

        weight = 1
        weight_list = []
        images_prob = []
        movies_prob = []

        grand_truth = 0
        grand_truth_list = []
        softmax = nn.Softmax()
        sigmoid = nn.Sigmoid()
        a_scores = []
        p_scores = []
        r_scores = []
        f_scores = []
        for batch_idx, (inputs1, targets,movie_name)  in enumerate(val_loader):

            if batch_idx == 0:
                current_movie = movie_name

            if use_cuda:
                inputs1, targets = inputs1.cuda(),targets.cuda()
            inputs1, targets = torch.autograd.Variable(inputs1, volatile=True), torch.autograd.Variable(targets)

            with torch.no_grad():
                att_outputs1, per_outputs1, _ = model(inputs1)

            if args.mode == "pn":  
                pos_prob1 = softmax(per_outputs1).detach().cpu().clone().numpy()[0][1]
            else:
                pos_prob1 = sigmoid(per_outputs1).detach().cpu().clone().numpy()[0][1]

            if current_movie == movie_name:
                images_prob.append(pos_prob1)
                weight_list.append(weight)
                weight += 1
                grand_truth = targets.detach().cpu().clone().numpy()[0]

            if current_movie != movie_name or len(frame) == batch_idx+1 :
                weighted_average=np.average(
                                    a= np.array(images_prob),       
                                    axis=None,                    
                                    weights=np.array(weight_list),  
                                    returned = False            
                                        )

                #print(str(current_movie[0])+" "+str(weighted_average))
                movies_prob.append(weighted_average)
                if grand_truth == 1:
                    grand_truth_list.append(1)
                else:
                    grand_truth_list.append(0)
                weight = 1 
                weight_list = []
                images_prob = []
                images_prob.append(pos_prob1)
                weight_list.append(weight)
                weight += 1
                
            current_movie = movie_name

        for threshold in thresholds:
            predict_label = []
            for prob in movies_prob:
                if prob >= threshold:
                    predict_label.append(1)
                else:
                    predict_label.append(0)

            a_scores.append(accuracy_score(grand_truth_list,predict_label))
            p_scores.append(precision_score(grand_truth_list,predict_label))
            r_scores.append(recall_score(grand_truth_list,predict_label))
            f_scores.append(f1_score(grand_truth_list,predict_label))
        a_all[i] = a_scores
        p_all[i] = p_scores
        r_all[i] = r_scores
        f_all[i] = f_scores

        roc_average_csv.append(roc_auc_score(grand_truth_list,movies_prob))
        pr_average_csv.append(average_precision_score(grand_truth_list,movies_prob))

    for i in range(len(thresholds)):
        a_average = 0
        p_average = 0
        r_average = 0
        f_average = 0
        for f in f_all:
            a_average += a_all[f][i]
            p_average += p_all[f][i]
            r_average += r_all[f][i]
            f_average += f_all[f][i]
        a_average /= args.seed_number
        p_average /= args.seed_number
        r_average /= args.seed_number
        f_average /= args.seed_number
        a_average_csv.append(a_average)
        p_average_csv.append(p_average)
        r_average_csv.append(r_average)
        f_average_csv.append(f_average)

    thresholds.insert(0,"threshold")
    a_average_csv.insert(0,"accuracy")
    p_average_csv.insert(0,"precision")
    r_average_csv.insert(0,"recall")
    f_average_csv.insert(0,"f-measure")

    with open(args.mode+'_test'+str(args.val_num)+'_weighted.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(thresholds)
        writer.writerow(a_average_csv)
        writer.writerow(p_average_csv)
        writer.writerow(r_average_csv)
        writer.writerow(f_average_csv)
        writer.writerow(pr_average_csv)
        writer.writerow(roc_average_csv)

if __name__ == '__main__':
    main()
