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


from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
"""
from pytext.config import ConfigBase
from pytext.config.component import Component, ComponentType
from pytext.utils import loss as loss_utils, precision
from pytext.utils.cuda import FloatTensor
"""

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--test-batch', default=1, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--layer', default=1, type=int, help='number of rnn layer')
parser.add_argument('--hidden', default=8, type=int, help='number of rnn unit')

#Device options
parser.add_argument('--gpu_id', default='0', type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--val_num', default='1', type=int,
                    help='id(s)')

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
        
        #if label[0] == -1:
        #    label[0] = 0

        return feature,length,label[0],npz_files[idx]

class VRNN(nn.Module):
    def __init__(self,input_size=2048,hidden_size=32,num_layers=1,max_seq_len=8,num_classes=2):
        super(VRNN, self).__init__()
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
pos_num = 0
neg_num = 0
for name in os.listdir("./feature/test"+str(args.val_num)):
    npz = np.load("./feature/test"+str(args.val_num)+"/"+name)
    feature = npz['arr_0']
    label = npz['arr_1']
    length = len(feature)

    if length > seq_len:
        seq_len = len(feature)
    if label[0] == 1:
        pos_num += 1
    if label[0] == -1:
        neg_num += 1



val_dataset = EGG(dir="./feature/test"+str(args.val_num),max_seq_len=seq_len,transform=transforms.Compose([
                                transforms.ToTensor(),
                                ]))

val_loader = torch.utils.data.DataLoader(
                                val_dataset,batch_size=args.test_batch, shuffle=False,
                                num_workers=args.workers, pin_memory=True)

model = VRNN(hidden_size=args.hidden,num_layers=args.layer)
model1 = torch.nn.DataParallel(model, device_ids=[int(args.gpu_id)])
model2 = torch.nn.DataParallel(model, device_ids=[int(args.gpu_id)])
model3 = torch.nn.DataParallel(model, device_ids=[int(args.gpu_id)])
model4 = torch.nn.DataParallel(model, device_ids=[int(args.gpu_id)])
model5 = torch.nn.DataParallel(model, device_ids=[int(args.gpu_id)])

checkpoint1 = torch.load("./checkpoints/variable_rnn"+str(args.val_num)+"_lr"+str(args.lr)+"_hidden"+str(args.hidden)+"_seed1.pth.tar", map_location="cuda:"+str(args.gpu_id))
checkpoint2 = torch.load("./checkpoints/variable_rnn"+str(args.val_num)+"_lr"+str(args.lr)+"_hidden"+str(args.hidden)+"_seed2.pth.tar", map_location="cuda:"+str(args.gpu_id))
checkpoint3 = torch.load("./checkpoints/variable_rnn"+str(args.val_num)+"_lr"+str(args.lr)+"_hidden"+str(args.hidden)+"_seed3.pth.tar", map_location="cuda:"+str(args.gpu_id))
checkpoint4 = torch.load("./checkpoints/variable_rnn"+str(args.val_num)+"_lr"+str(args.lr)+"_hidden"+str(args.hidden)+"_seed4.pth.tar", map_location="cuda:"+str(args.gpu_id))
checkpoint5 = torch.load("./checkpoints/variable_rnn"+str(args.val_num)+"_lr"+str(args.lr)+"_hidden"+str(args.hidden)+"_seed5.pth.tar", map_location="cuda:"+str(args.gpu_id))

model1.load_state_dict(checkpoint1['state_dict'])
model2.load_state_dict(checkpoint2['state_dict'])
model3.load_state_dict(checkpoint3['state_dict'])
model4.load_state_dict(checkpoint4['state_dict'])
model5.load_state_dict(checkpoint5['state_dict'])

model1.eval()
model2.eval()
model3.eval()
model4.eval()
model5.eval()

probs = []
grand_truth_list = []
sigmoid = nn.Sigmoid()
for batch_idx, (feature,length,labels,movie_name) in enumerate(val_loader):       

    feature, labels = feature.to(args.gpu_id), labels.to(args.gpu_id)

    output1 = sigmoid(model1(feature,length)[0][1])
    output2 = sigmoid(model2(feature,length)[0][1])
    output3 = sigmoid(model3(feature,length)[0][1])
    output4 = sigmoid(model4(feature,length)[0][1])
    output5 = sigmoid(model5(feature,length)[0][1])

    output = (output1 + output2 + output3 + output4 + output5)/5

    probs.append(output)

    grand_truth_list.append(labels.detach().cpu().clone().numpy()[0])

thresholds = [0.1,0.2,0.3,0.4,0.5]
for threshold in thresholds:
    print("threshold:{}".format(threshold))
    predict_label = []
    for prob in probs:
        if prob > threshold:
            predict_label.append(1)
        else:
            predict_label.append(-1)

    F_score = f1_score(grand_truth_list, predict_label)
    print("F_score:{}".format(F_score))
    accuracy = accuracy_score(grand_truth_list,predict_label)
    print("Accuracy:{}".format(accuracy))
    precision = precision_score(grand_truth_list, predict_label)
    print("precision:{}".format(precision))
    print(confusion_matrix(grand_truth_list,predict_label, labels=[1, -1]))

print("auc:{}".format(roc_auc_score(grand_truth_list,probs))) 

