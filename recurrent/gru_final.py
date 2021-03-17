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
from pytext.config import ConfigBase
from pytext.config.component import Component, ComponentType
from pytext.utils import loss as loss_utils, precision
from pytext.utils.cuda import FloatTensor

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
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--train-batch', default=16, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=1, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--layer', default=1, type=int, help='number of rnn layer')
parser.add_argument('--hidden', default=64, type=int, help='number of rnn rnn unit')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--beta', '-B', default=0.5, type=float,
                    help='Beta parameter of nnPU')
parser.add_argument('--gamma_pu', '-G', default=0.05, type=float,
                    help='Gamma parameter of nnPU')

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


class Loss(Component):
    """Base class for loss functions"""

    __COMPONENT_TYPE__ = ComponentType.LOSS

    def __init__(self, config=None, *args, **kwargs):
        super().__init__(config)

    def __call__(self, logit, targets, reduce=True):
        raise NotImplementedError


class Config(ConfigBase):
    precision_range_lower = 0.0
    precision_range_upper = 1.0
    num_classes = 2
    num_anchors = 20

class AUCPRHingeLoss(nn.Module, Loss):
    """area under the precision-recall curve loss,
    Reference: "Scalable Learning of Non-Decomposable Objectives", Section 5 \
    TensorFlow Implementation: \
    https://github.com/tensorflow/models/tree/master/research/global_objectives\
    """

    class Config(ConfigBase):
        """
        Attributes:
            precision_range_lower (float): the lower range of precision values over
                which to compute AUC. Must be nonnegative, `\leq precision_range_upper`,
                and `leq 1.0`.
            precision_range_upper (float): the upper range of precision values over
                which to compute AUC. Must be nonnegative, `\geq precision_range_lower`,
                and `leq 1.0`.
            num_classes (int): number of classes(aka labels)
            num_anchors (int): The number of grid points used to approximate the
                Riemann sum.
        
        
        precision_range_lower: float = 0.0
        precision_range_upper: float = 1.0
        num_classes: int = 1
        num_anchors: int = 20
        """
        precision_range_lower = 0.0
        precision_range_upper = 1.0
        num_classes = 2
        num_anchors = 20

    def __init__(self, config, weights=None, *args, **kwargs):
        """Args:
            config: Config containing `precision_range_lower`, `precision_range_upper`,
                `num_classes`, `num_anchors`
        """
        nn.Module.__init__(self)
        Loss.__init__(self, config)

        self.num_classes = self.config.num_classes
        self.num_anchors = self.config.num_anchors
        self.precision_range = (
            self.config.precision_range_lower,
            self.config.precision_range_upper,
        )

        # Create precision anchor values and distance between anchors.
        # coresponding to [alpha_t] and [delta_t] in the paper.
        # precision_values: 1D `Tensor` of shape [K], where `K = num_anchors`
        # delta: Scalar (since we use equal distance between anchors)
        self.precision_values, self.delta = loss_utils.range_to_anchors_and_delta(
            self.precision_range, self.num_anchors
        )

        # notation is [b_k] in paper, Parameter of shape [C, K]
        # where `C = number of classes` `K = num_anchors`
        self.biases = nn.Parameter(
            FloatTensor(self.config.num_classes, self.config.num_anchors).zero_()
        )
        self.lambdas = nn.Parameter(
            FloatTensor(self.config.num_classes, self.config.num_anchors).data.fill_(
                1.0
            )
        )

    def forward(self, logits, targets, reduce=True, size_average=True, weights=None):
        """
        Args:
            logits: Variable :math:`(N, C)` where `C = number of classes`
            targets: Variable :math:`(N)` where each value is
                `0 <= targets[i] <= C-1`
            weights: Coefficients for the loss. Must be a `Tensor` of shape
                [N] or [N, C], where `N = batch_size`, `C = number of classes`.
            size_average (bool, optional): By default, the losses are averaged
                    over observations for each minibatch. However, if the field
                    sizeAverage is set to False, the losses are instead summed
                    for each minibatch. Default: ``True``
            reduce (bool, optional): By default, the losses are averaged or summed over
                observations for each minibatch depending on size_average. When reduce
                is False, returns a loss per input/target element instead and ignores
                size_average. Default: True
        """
        C = 1 if logits.dim() == 1 else logits.size(1)

        if self.num_classes != C:
            raise ValueError(
                "num classes is %d while logits width is %d" % (self.num_classes, C)
            )

        #logits = self.softmax(logits)

        labels, weights = AUCPRHingeLoss._prepare_labels_weights(
            logits, targets, weights=weights
        )


        # Lagrange multipliers
        # Lagrange multipliers are required to be nonnegative.
        # Their gradient is reversed so that they are maximized
        # (rather than minimized) by the optimizer.
        # 1D `Tensor` of shape [K], where `K = num_anchors`
        lambdas = loss_utils.lagrange_multiplier(self.lambdas.to(args.gpu_id) )

        # print("lambdas: {}".format(lambdas))
        # A `Tensor` of Shape [N, C, K]
        positive_loss, negative_loss = loss_utils.weighted_pu_hinge_loss(
            labels.unsqueeze(-1),
            logits.unsqueeze(-1) - self.biases,
            positive_weights=1.0 + lambdas.to(args.gpu_id)  * (1.0 - self.precision_values.to(args.gpu_id) ),
            negative_weights=lambdas * self.precision_values.to(args.gpu_id) ,
        )

        # 1D tensor of shape [C]
        class_priors = loss_utils.build_class_priors(labels,args.gpu_id, weights=weights)
        
        # lambda_term: Tensor[C, K]
        # according to paper, lambda_term = lambda * (1 - precision) * |Y^+|
        # where |Y^+| is number of postive examples = N * class_priors
        lambda_term = class_priors.unsqueeze(-1) * (
            lambdas * (1.0 - self.precision_values.to(args.gpu_id)))

        positive_per_anchor_loss = weights.unsqueeze(-1).to(args.gpu_id)  * positive_loss.to(args.gpu_id) 
        negative_per_anchor_loss = weights.unsqueeze(-1).to(args.gpu_id)  * negative_loss.to(args.gpu_id)
        
        # Riemann sum over anchors, and normalized by precision range
        # loss: Tensor[N, C]
        positive_loss = positive_per_anchor_loss.sum(2) * self.delta
        negative_loss = negative_per_anchor_loss.sum(2) * self.delta
        lambda_term = lambda_term * self.delta

        positive_loss /= self.precision_range[1] - self.precision_range[0]
        negative_loss /= self.precision_range[1] - self.precision_range[0]
        lambda_term /= self.precision_range[1] - self.precision_range[0]

        if not reduce:
            return positive_loss,negative_loss,lambda_term
        elif size_average:
            return positive_loss.mean(),negative_loss.mean(),lambda_term.mean()
        else:
            return positive_loss.sum(),negative_loss.sum(),lambda_term.sum()

    @staticmethod
    def _prepare_labels_weights(logits, targets, weights=None):
        """
        Args:
            logits: Variable :math:`(N, C)` where `C = number of classes`
            targets: Variable :math:`(N)` where each value is
                `0 <= targets[i] <= C-1`
            weights: Coefficients for the loss. Must be a `Tensor` of shape
                [N] or [N, C], where `N = batch_size`, `C = number of classes`.
        Returns:
            labels: Tensor of shape [N, C], one-hot representation
            weights: Tensor of shape broadcastable to labels
        """
        N, C = logits.size()

        # Converts targets to one-hot representation. Dim: [N, C]
        labels = FloatTensor(N, C).to(args.gpu_id).zero_().scatter(1, targets.unsqueeze(1), 1)

        if weights is None:
            weights = FloatTensor(N).to(args.gpu_id).fill_(1.0)

        if weights.dim() == 1:
            weights.unsqueeze_(-1)

        return labels, weights

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

        return feature,length,label[0]

class VRNN(nn.Module):
    def __init__(self,input_size=2048,hidden_size=512,num_layers=2,max_seq_len=8,num_classes=2):
        super(VRNN, self).__init__()
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

def train(train_loader, model,criterion_auc, optimizer, epoch, use_cuda):
    bar = Bar('Processing', max=len(train_loader))
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    for batch_idx, (feature,length,labels) in enumerate(train_loader):   

        feature, labels = feature.to(args.gpu_id), labels.to(args.gpu_id)

        output = model(feature,length)
        positive_loss, negative_loss, lambda_term = criterion_auc(output, labels)

        if negative_loss < -1*args.beta:
            auc_loss = - args.gamma_pu * negative_loss

        else:
            auc_loss = positive_loss + negative_loss - lambda_term

        optimizer.zero_grad()
        auc_loss.backward()
        optimizer.step()     

        pred=[]
        output = torch.chunk(output,2,dim=1)
        for i in range(len(output[1].detach().cpu().clone().numpy())):
            if output[1].detach().cpu().clone().numpy()[i] >0:
                pred.append(1)
            else:
                pred.append(0)

        all_count = 0
        acc_count = 0
        for i  in range(len(labels.detach().cpu().clone().numpy())):
            if labels.detach().cpu().clone().numpy()[i] == pred[i]:
                acc_count += 1
            all_count += 1

        prec1 = acc_count/all_count*100

        top1.update(prec1, feature.size(0))
        losses.update(auc_loss, feature.size(0))

        # plot progress
        bar.suffix  = '({batch}/{size}) | Loss: {per_loss:.4f} | top1: {top1: .4f} '.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    per_loss=losses.avg,
                    top1=top1.avg
                    )
        bar.next()
    bar.finish()
    return losses.avg, top1.avg

def val(val_loader, model,criterion_auc, use_cuda):
    model.eval()
    bar = Bar('Processing', max=len(val_loader))
    losses = AverageMeter()
    top1 = AverageMeter()
    for batch_idx, (feature,length,labels) in enumerate(val_loader):   

        feature, labels = feature.to(args.gpu_id), labels.to(args.gpu_id)

        with torch.no_grad():
            output = model(feature,length)
            positive_loss, negative_loss, lambda_term = criterion_auc(output, labels)

        auc_loss = positive_loss + negative_loss - lambda_term

        pred=[]
        output = torch.chunk(output,2,dim=1)
        for i in range(len(output[1].detach().cpu().clone().numpy())):
            if output[1].detach().cpu().clone().numpy()[i] >0:
                pred.append(1)
            else:
                pred.append(0)

        all_count = 0
        acc_count = 0
        for i  in range(len(labels.detach().cpu().clone().numpy())):
            if labels.detach().cpu().clone().numpy()[i] == pred[i]:
                acc_count += 1
            all_count += 1

        prec1 = acc_count/all_count*100

        top1.update(prec1, feature.size(0))
        losses.update(auc_loss, feature.size(0))

        # plot progress
        bar.suffix  = '({batch}/{size}) | Loss: {per_loss:.4f} | top1: {top1: .4f} '.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    per_loss=losses.avg,
                    top1=top1.avg
                    )
        bar.next()
    bar.finish()
    return losses.avg, top1.avg

def save_checkpoint(state,checkpoint='checkpoint', filename="gru"+str(args.val_num)+"_seed"+str(args.manualSeed)+".pth.tar"):
    filepath = os.path.join(checkpoint, filename)   
    torch.save(state, filepath)

file_name = "gru_final"+str(args.val_num)+"_seed"+str(args.manualSeed)
logger1 = Logger('train-'+file_name+'.txt')
logger2 = Logger('val-'+file_name+'.txt')
logger1.set_names([ 'Train Loss' ,'Train Acc'])
logger2.set_names([ 'Val Loss' ,'Val Acc'])

seq_len = 0
for name in os.listdir("./feature/train"+str(args.val_num)):
    npz = np.load("./feature/train"+str(args.val_num)+"/"+name)
    feature = npz['arr_0']
    label = npz['arr_1']
    length = len(feature)

    if length > seq_len:
        seq_len = len(feature)

train_dataset = EGG(dir="./feature/train"+str(args.val_num),max_seq_len=seq_len,transform=transforms.Compose([
                                transforms.ToTensor(),
                                ]))

train_loader = torch.utils.data.DataLoader(
                                train_dataset,batch_size=args.train_batch, shuffle=True,
                                num_workers=args.workers, pin_memory=True)

val_dataset = EGG(dir="./feature/test"+str(args.val_num),max_seq_len=seq_len,transform=transforms.Compose([
                                transforms.ToTensor(),
                                ]))

val_loader = torch.utils.data.DataLoader(
                                val_dataset,batch_size=args.test_batch, shuffle=False,
                                num_workers=args.workers, pin_memory=True)

model = VRNN(max_seq_len=seq_len,hidden_size=args.hidden,num_layers=args.layer)
model = torch.nn.DataParallel(model, device_ids=[int(args.gpu_id)])
fig_file=Config()
criterion_auc = AUCPRHingeLoss(config=fig_file).to(int(args.gpu_id))
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Train and val
for epoch in range(0, args.epochs):

    auc_loss,top1 = train(train_loader, model,criterion_auc , optimizer, epoch, use_cuda)
    auc_loss_val,top1_val = val(val_loader, model,criterion_auc , use_cuda)

    # append logger file
    logger1.append([auc_loss,top1])
    logger2.append([auc_loss_val,top1_val])

    # save model
    save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                    },checkpoint="checkpoint")

logger1.close()
logger1.plot()
logger2.close()
logger2.plot()




