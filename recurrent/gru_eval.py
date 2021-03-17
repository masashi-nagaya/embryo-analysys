from __future__ import print_function

import argparse
import os
import shutil
import time
import numpy as np
import random
import pickle
import os
# import cv2

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Dataset
from os import path
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
import csv


from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
"""
from pytext.config import ConfigBase
from pytext.config.component import Component, ComponentType
from pytext.utils import loss as loss_utils, precision
from pytext.utils.cuda import FloatTensor
"""

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--test-batch', default=1, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--layer', default=1, type=int, help='number of rnn layer')
parser.add_argument('--hidden', default=64, type=int, help='number of rnn unit')

parser.add_argument('--beta', default=2, type=float, help='number of rnn layer')
parser.add_argument('--epochs', default=10, type=int, help='number of rnn unit')

parser.add_argument('--gpu_id', default='0', type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--val_num', default='1', type=int,
                    help='id(s)')

parser.add_argument('--seed_number', default=5, type=int,
                    help='seed Number of times')

parser.add_argument('--arch', default='final', type=str,
                    choices=["final","temporal"],
                    help='architecture type')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

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


class EGG(Dataset):
    def __init__(self, dir,max_seq_len, transform=None):
        self.dir = dir
        self.max_seq_len = max_seq_len
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.dir))

    def __getitem__(self, idx):
        npz_files = os.listdir(self.dir)
        npz = np.load(self.dir+"/"+npz_files[idx])
        feature = npz['arr_0']
        label = npz['arr_1']
        feature = feature.reshape([-1,2048])
        length = len(feature)

        feature = np.pad(feature,[(0,self.max_seq_len-len(feature)),(0,0)],"constant")
        
        if label[0] == -1:
            label[0] = 0

        return feature,length,label[0],npz_files[idx]

class VRNN_final(nn.Module):
    def __init__(self,input_size=2048,hidden_size=512,num_layers=2,max_seq_len=8,num_classes=2):
        super(VRNN_final, self).__init__()
        self.input_size = input_size
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.output_size = num_classes
        self.relu = nn.ReLU(inplace=True)
        self.rnn = nn.GRU(input_size,hidden_size,num_layers,batch_first=True) 
        self.dropout = nn.Dropout()

        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax()        

    def forward(self,x,lengths):

        batch_size, seq_len, feature_len = x.size()
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True,enforce_sorted=False)
        output, _ = self.rnn(x)
        output, lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=False)

        seq_len,batch_size,feature_len = output.size()
        output = output.view(batch_size*seq_len, self.hidden_size)
        adjusted_lengths = [(l-1)*batch_size + i for i,l in enumerate(lengths)]

        lengthTensor = torch.tensor(adjusted_lengths, dtype=torch.int64)
        lengthTensor = lengthTensor.to(int(args.gpu_id))

        output = output.index_select(0,lengthTensor)
        output = output.view(batch_size,self.hidden_size)

        output = self.dropout(output)
        output = self.fc(output)
        return output

class VRNN_temporal(nn.Module):
    def __init__(self,input_size=2048,hidden_size=32,num_layers=1,max_seq_len=8,num_classes=2):
        super(VRNN_temporal, self).__init__()
        self.input_size = input_size
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.output_size = num_classes
        self.relu = nn.ReLU(inplace=True)
        self.rnn = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True) 
        self.dropout = nn.Dropout()

        self.fc = nn.Linear(hidden_size*2, num_classes)      

    def forward(self,x,lengths):
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True,enforce_sorted=False)
        x, _ = self.rnn(x)
        
        x, seq_len = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        avg_pool = F.adaptive_avg_pool1d(x.permute(0,2,1),1).view(x.size(0),-1)
        max_pool = F.adaptive_max_pool1d(x.permute(0,2,1),1).view(x.size(0),-1)

        pooled_feature = torch.cat([avg_pool,max_pool],dim=1)

        output = self.dropout(pooled_feature)
        out = self.fc(output)
        
        return out

seq_len = 0

for name in os.listdir("./feature/test"+str(args.val_num)):
    npz = np.load("./feature/test"+str(args.val_num)+"/"+name)
    feature = npz['arr_0']
    label = npz['arr_1']
    length = len(feature)

    if length > seq_len:
        seq_len = len(feature)

val_dataset = EGG(dir="./feature/test"+str(args.val_num),max_seq_len=seq_len,transform=transforms.Compose([
                                transforms.ToTensor(),
                                ]))

val_loader = torch.utils.data.DataLoader(
                                val_dataset,batch_size=args.test_batch, shuffle=False,
                                num_workers=args.workers, pin_memory=True)

if args.arch == "final":
    model = VRNN_final(hidden_size=args.hidden,num_layers=args.layer)
elif args.arch == "temporal":
    model = VRNN_temporal(hidden_size=args.hidden,num_layers=args.layer) 

model = torch.nn.DataParallel(model, device_ids=[int(args.gpu_id)])


model.eval()

probs = []
grand_truth_list = []
sigmoid = nn.Sigmoid()

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

    a_scores = []
    p_scores = []
    r_scores = []
    f_scores = []
    for batch_idx, (feature,length,labels,movie_name) in enumerate(val_loader):       

        feature, labels = feature.to(args.gpu_id), labels.to(args.gpu_id)
        output1 = sigmoid(model(feature,length)[0][1])

        probs.append(output1)
        grand_truth_list.append(labels.detach().cpu().clone().numpy()[0])

    for threshold in thresholds:
        predict_label = []
        for prob in probs:
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

    roc_average_csv.append(roc_auc_score(grand_truth_list,probs))
    pr_average_csv.append(average_precision_score(grand_truth_list,probs))

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

with open('test'+str(args.val_num)+'.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(thresholds)
    writer.writerow(a_average_csv)
    writer.writerow(p_average_csv)
    writer.writerow(r_average_csv)
    writer.writerow(f_average_csv)
    writer.writerow(pr_average_csv)
    writer.writerow(roc_average_csv)


