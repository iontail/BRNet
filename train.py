# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import random
import time
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchmetrics.functional import structural_similarity_index_measure as ssim
from tqdm import tqdm
import wandb

from models.data.config import cfg
from models.data.widerface import WIDERDetection, detection_collate
from models.factory import build_net, basenet_factory
from models.modules.enhancer import RetinexNet
from PIL import Image
from BRTrainer import BR_Trainer, adjust_learning_rate

parser = argparse.ArgumentParser(
    description='DSFD face Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--batch_size',
                    default=4, type=int,
                    help='Batch size for training')
parser.add_argument('--model',
                    default='dark', type=str,
                    choices=['dark', 'vgg', 'resnet50', 'resnet101', 'resnet152'],
                    help='model for training')
parser.add_argument('--resume',
                    default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers',
                    default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda',
                    default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate',
                    default=5e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',
                    default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma',
                    default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--multigpu',
                    default=True, type=bool,
                    help='Use mutil Gpu training')
parser.add_argument('--save_folder',
                    default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--local_rank',
                    type=int,
                    help='local rank for dist')
parser.add_argument('--use_wandb',
                    default = True, type = bool,
                    help='Whether using wandb log')

args = parser.parse_args()
global local_rank
local_rank = args.local_rank

if 'LOCAL_RANK' not in os.environ:
    os.environ['LOCAL_RANK'] = str(args.local_rank)

if torch.cuda.is_available():
    if args.cuda:
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        import torch.distributed as dist

        gpu_num = torch.cuda.device_count()
        if local_rank == 0:
            print('Using {} gpus'.format(gpu_num))
        rank = int(os.environ['RANK'])
        torch.cuda.set_device(rank % gpu_num)
        dist.init_process_group('nccl')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

save_folder = os.path.join(args.save_folder, args.model)
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

train_dataset = WIDERDetection(cfg.FACE.TRAIN_FILE, mode='train')

val_dataset = WIDERDetection(cfg.FACE.VAL_FILE, mode='val')
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
train_loader = data.DataLoader(train_dataset, args.batch_size,
                               num_workers=args.num_workers,
                               collate_fn=detection_collate,
                               sampler=train_sampler,
                               pin_memory=True)
val_batchsize = args.batch_size
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=True)
val_loader = data.DataLoader(val_dataset, val_batchsize,
                             num_workers=0,
                             collate_fn=detection_collate,
                             sampler=val_sampler,
                             pin_memory=True)


min_loss = np.inf

if args.use_wandb:
    wandb.init(project = cfg.MODEL_NAME)
    wandb.config = {
        'learning_rate': args.lr,
        'epooch': cfg.EPOCHES,
        'batch_size': args.batch_size
    }

def train():
    
    per_epoch_size = len(train_dataset) // (args.batch_size * torch.cuda.device_count())
    start_epoch = 0
    iteration = 0
    step_index = 0

    basenet = basenet_factory(args.model)
    dsfd_net = build_net('train', cfg.NUM_CLASSES, args.model)
    net = dsfd_net
    net_enh = RetinexNet()
    net_enh.load_state_dict(torch.load(args.save_folder + 'decomp.pth'))

    if args.resume:
        if local_rank == 0:
            print('Resuming training, loading {}...'.format(args.resume))
        start_epoch = net.load_weights(args.resume)
        iteration = start_epoch * per_epoch_size
    else:
        # not use pretrained model
        # because we use our own model, not VGG
        """
        base_weights = torch.load(args.save_folder + basenet)
        if local_rank == 0:
            print('Load base network {}'.format(args.save_folder + basenet))
        if args.model == 'vgg' or args.model == 'dark':
            net.vgg.load_state_dict(base_weights)
        else:
            net.resnet.load_state_dict(base_weights)
        """


    if not args.resume:
        if local_rank == 0:
            print('Initializing weights...')
        net.extras.apply(net.weights_init)
        net.fpn_topdown.apply(net.weights_init)
        net.fpn_latlayer.apply(net.weights_init)
        net.fpn_fem.apply(net.weights_init)
        net.loc_pal1.apply(net.weights_init)
        net.conf_pal1.apply(net.weights_init)
        net.loc_pal2.apply(net.weights_init)
        net.conf_pal2.apply(net.weights_init)
        net.ref.apply(net.weights_init)

    # Scaling the lr
    lr = args.lr * np.round(np.sqrt(args.batch_size / 4 * torch.cuda.device_count()),4) ##?????????????
    param_group = []
    param_group += [{'params': dsfd_net.brnet.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.extras.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.fpn_topdown.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.fpn_latlayer.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.fpn_fem.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.loc_pal1.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.conf_pal1.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.loc_pal2.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.conf_pal2.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.brnet.ref.parameters(), 'lr': lr / 10.}]
    param_group += [{'params': dsfd_net.brnet.dark.parameters(), 'lr': lr / 10.}]

    optimizer = optim.Adamw(param_group, lr=lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    if args.cuda:
        if args.multigpu:
            net = torch.nn.parallel.DistributedDataParallel(net.cuda(), find_unused_parameters=True)
            net_enh = torch.nn.parallel.DistributedDataParallel(net_enh.cuda())
        # net = net.cuda()
        random.seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        torch.cuda.manual_seed(cfg.SEED)
        cudnn.benchmark = False
        

    
    
    if local_rank == 0:
        print('Loading wider dataset...')
        print('Using the specified args:')
        print(args)

    for step in cfg.LR_STEPS:
        # resume training의 경우에는 iteration에 맞게 lr 조정
        # lr decay by 1/10 in [10000*2,12500*2,15000*2] iterations
        if iteration > step:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

    net_enh.eval()
    net.train()
    
    trainer = BR_Trainer(net, net_enh, train_loader, val_loader, optimizer,
                         cfg, args, epochs=cfg.EPOCHES, start_epoch = start_epoch,
                         eval_steps=5000, checkpoint_dir=save_folder)
    
    trainer.train()
    
    


if __name__ == '__main__':
    train()
