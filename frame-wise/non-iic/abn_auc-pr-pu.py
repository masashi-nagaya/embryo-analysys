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

from pytext.config import ConfigBase
from pytext.config.component import Component, ComponentType
from pytext.utils import loss as loss_utils, precision
from pytext.utils.cuda import FloatTensor

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
    
parser.add_argument('--mse_loss',  default=1, type=float,
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
parser.add_argument('--gpu_id', default='0', type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--beta', '-B', default=4, type=float,
                    help='Beta parameter of nnPU')
parser.add_argument('--gamma_pu', '-G', default=0.1, type=float,
                    help='Gamma parameter of nnPU')
parser.add_argument('--val_num', default=1, type=int,
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

        elif int(self.frame.iloc[idx, 0].split()[1]) == -5 : 
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
                                val_dataset,batch_size=args.test_batch, shuffle=True,
                                num_workers=args.workers, pin_memory=True)

    # create model
    model = models.__dict__[args.arch]()
    model = torch.nn.DataParallel(model, device_ids=[int(args.gpu_id)])

    cudnn.benchmark = True

    fig_file=Config()
    criterion_auc = AUCPRHingeLoss(config=fig_file).to(args.gpu_id) 
    criterion_mse = nn.MSELoss().to(args.gpu_id) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Resume
    title = 'ImageNet-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        print(args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        logger1 = Logger("auc-pr-pu-train"+str(args.val_num)+"_mse"+str(args.mse_loss)+"_seed"+str(args.manualSeed)+'.txt', title=title)
        logger1.set_names(['auc_loss_att','auc_loss_per','mse_loss','accuracy'])
        logger2 = Logger("auc-pr-pu-val"+str(args.val_num)+"_mse"+str(args.mse_loss)+"_seed"+str(args.manualSeed)+'.txt', title=title)
        logger2.set_names(['auc_loss_att','auc_loss_per','accuracy'])

    model.module.att_conv2 = nn.Conv2d(1000, 2, kernel_size=1, padding=0,bias=False) 
    model.module.fc = nn.Linear(2048, 2)
    model = model.to(args.gpu_id)   

    # Train and val
    for epoch in range(start_epoch, args.epochs):

        auc_losses_att, auc_losses_per , mse_losses, train_acc = train(train_loader, model ,criterion_auc,criterion_mse, optimizer, epoch, use_cuda)
        auc_losses_att_val, auc_losses_per_val , acc_val = val(val_loader, model,criterion_auc , use_cuda)        

        logger1.append([auc_losses_att, auc_losses_per ,mse_losses ,train_acc])
        logger2.append([auc_losses_att_val, auc_losses_per_val ,acc_val])

        # save model
        save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, checkpoint=args.checkpoint)
        
    logger1.close()
    logger1.plot()
    logger2.close()
    logger2.plot()

def train(train_loader, model, criterion_auc,criterion_mse, optimizer, epoch, use_cuda):

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    mse_losses = AverageMeter()
    auc_losses_per = AverageMeter()
    auc_losses_att = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    sigmoid = nn.Sigmoid()
    
    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs1, inputs2, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs1, inputs2, targets = inputs1.to(args.gpu_id) ,inputs2.to(args.gpu_id) , targets.to(args.gpu_id)
        inputs1,inputs2, targets = torch.autograd.Variable(inputs1),torch.autograd.Variable(inputs2), torch.autograd.Variable(targets)

        # compute output
        att_outputs1, outputs1, _  = model(inputs1)
        att_outputs2, outputs2, _  = model(inputs2)

        mse_loss_att = criterion_mse(att_outputs2,att_outputs1.detach())
        mse_loss_per = criterion_mse(outputs1,outputs2.detach())

        positive_loss_att, negative_loss_att, lambda_term_att = criterion_auc(att_outputs1, targets)
        positive_loss_per, negative_loss_per, lambda_term_per = criterion_auc(outputs1, targets)

        auc_loss_att = positive_loss_att + negative_loss_att - lambda_term_att
        auc_loss_per = positive_loss_per + negative_loss_per - lambda_term_per

        if negative_loss_att < -1*args.beta and negative_loss_per < -1*args.beta :
            auc_loss = - args.gamma_pu * negative_loss_att - args.gamma_pu * negative_loss_per
            #print("both negative")

        elif negative_loss_att < -1*args.beta:
            auc_loss = - args.gamma_pu * negative_loss_att + auc_loss_per
            #print("attention negative")

        elif negative_loss_per < -1*args.beta:
            auc_loss =  auc_loss_att - args.gamma_pu * negative_loss_per
            #print("per negative")

        else:
            auc_loss = auc_loss_att + auc_loss_per

        loss = auc_loss + args.mse_loss*mse_loss_att + args.mse_loss*mse_loss_per

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs1 = torch.chunk(outputs1,2,dim=1)
        att_outputs1 = torch.chunk(att_outputs1,2,dim=1)

        # measure accuracy and record loss
        outputs1 = sigmoid(outputs1[1].data)
    
        all_count = 0
        acc_count = 0
        for i  in range(len(targets.detach().cpu().clone().numpy())):
            if outputs1.detach().cpu().clone().numpy()[i] >0.5:
                if targets.detach().cpu().clone().numpy()[i] == 1:
                    acc_count += 1
            if outputs1.detach().cpu().clone().numpy()[i] <0.5:
                if targets.detach().cpu().clone().numpy()[i] == 0:
                    acc_count += 1            

            all_count += 1

        prec1 = acc_count/all_count*100

        auc_losses_per.update(auc_loss_per, inputs1.size(0))
        auc_losses_att.update(auc_loss_att, inputs1.size(0))
        mse_losses.update(mse_loss_per, inputs1.size(0))
        top1.update(prec1, inputs1.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) |Total: {total:} | ETA: {eta:} | auc_Loss: {auc_loss:.4f}|mse_Loss:{mse_loss:.4f}| top1: {top1: .4f} '.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    auc_loss=auc_losses_per.avg,
                    mse_loss=mse_losses.avg,
                    top1=top1.avg
                    )
        bar.next()
    bar.finish()
    return (auc_losses_att.avg ,auc_losses_per.avg  , mse_losses.avg, top1.avg)

def val(val_loader, model, criterion_auc, use_cuda):

    # switch to train mode
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    auc_losses_per = AverageMeter()
    auc_losses_att = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    sigmoid = nn.Sigmoid()
    
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs1, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs1, targets = inputs1.to(args.gpu_id) , targets.to(args.gpu_id)
        inputs1, targets = torch.autograd.Variable(inputs1), torch.autograd.Variable(targets)

        # compute output
        with torch.no_grad():
            att_outputs1, outputs1, _  = model(inputs1)

        positive_loss_att, negative_loss_att, lambda_term_att = criterion_auc(att_outputs1, targets)
        positive_loss_per, negative_loss_per, lambda_term_per = criterion_auc(outputs1, targets)

        auc_loss_att = positive_loss_att + negative_loss_att - lambda_term_att
        auc_loss_per = positive_loss_per + negative_loss_per - lambda_term_per

        outputs1 = torch.chunk(outputs1,2,dim=1)
        att_outputs1 = torch.chunk(att_outputs1,2,dim=1)

        # measure accuracy and record loss
        outputs1 = sigmoid(outputs1[1].data)
    
        all_count = 0
        acc_count = 0
        for i  in range(len(targets.detach().cpu().clone().numpy())):
            if outputs1.detach().cpu().clone().numpy()[i] >0.5:
                if targets.detach().cpu().clone().numpy()[i] == 1:
                    acc_count += 1
            if outputs1.detach().cpu().clone().numpy()[i] <0.5:
                if targets.detach().cpu().clone().numpy()[i] == 0:
                    acc_count += 1            
            all_count += 1

        prec1 = acc_count/all_count*100

        auc_losses_per.update(auc_loss_per, inputs1.size(0))
        auc_losses_att.update(auc_loss_att, inputs1.size(0))
        top1.update(prec1, inputs1.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) |Total: {total:} | ETA: {eta:} | auc_Loss: {auc_loss:.4f}| top1: {top1: .4f} '.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    auc_loss=auc_losses_per.avg,
                    top1=top1.avg
                    )
        bar.next()
    bar.finish()
    return (auc_losses_att.avg ,auc_losses_per.avg  ,top1.avg)

def save_checkpoint(state , checkpoint='checkpoint',  filename="auc-pr-pu"+str(args.val_num)+"_seed"+str(args.manualSeed)+'.pth.tar'):
    filepath = os.path.join(checkpoint, filename)   
    torch.save(state, filepath)

if __name__ == '__main__':
    main()


