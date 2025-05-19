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
from models.factory import build_net
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
                    default=torch.cuda.is_available(), type=bool,
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

local_rank = int(os.environ['LOCAL_RANK'])  # GPU 번호
device = torch.device(f"cuda:{local_rank}")  # 이걸 기반으로 .to(device)에서 사용

if torch.cuda.is_available():
    if args.cuda:
        import torch.distributed as dist

        gpu_num = torch.cuda.device_count()
        if local_rank == 0:
            print('Using {} gpus'.format(gpu_num))

        torch.cuda.set_device(local_rank)  # 여기 핵심! rank가 아닌 local_rank로 GPU 할당
        dist.init_process_group(backend='nccl')  # DDP 초기화
        
    else:
        print("WARNING: CUDA is available but not used. Use --cuda for better performance.")
else:
    print("WARNING: CUDA is not available. Training will be slow.")

save_folder = os.path.join(args.save_folder, args.model)
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

train_dataset = WIDERDetection(cfg.FACE.TRAIN_FILE, mode='train')

val_dataset = WIDERDetection(cfg.FACE.VAL_FILE, mode='val')

# for split the dataset matching the local rank
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
train_loader = data.DataLoader(train_dataset, args.batch_size,
                               num_workers=args.num_workers,
                               shuffle = False,
                               collate_fn=detection_collate,
                               sampler=train_sampler,
                               pin_memory=True)
val_batchsize = args.batch_size
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
val_loader = data.DataLoader(val_dataset, val_batchsize,
                             num_workers=0,
                             shuffle = False,
                             collate_fn=detection_collate,
                             sampler=val_sampler,
                             pin_memory=True)


min_loss = np.inf

if args.use_wandb and local_rank == 0: # avoid overlap logging
    wandb.init(project = cfg.MODEL_NAME)
    wandb.config = {
        'learning_rate': args.lr,
        'epoch': cfg.EPOCHES,
        'batch_size': args.batch_size
    }

def train():
    
    
    #per_epoch_size = len(train_dataset) // (args.batch_size * torch.cuda.device_count())
    # sampler사용하기 떄문에 아래로 고쳐야함
    per_epoch_size = len(train_sampler) // args.batch_size

    start_epoch = 0
    iteration = 0
    step_index = 0

    net = build_net("train", cfg.NUM_CLASSES)
    net_enh = RetinexNet()
    net_enh.load_state_dict(torch.load(args.save_folder + 'decomp.pth'))
    optimizer_state_dict = None

    if args.resume:
        # simply print 0th gpu resume
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
        net.brnet.apply(net.weights_init)


    # Scaling the lr
    # batch size가 4일때의 lr를 기준으로 scaling
    # 왜냐하면 저자들은 4로 학습했기 때문
    lr = args.lr * np.round(np.sqrt(args.batch_size / 4 * torch.cuda.device_count()),4)
    param_group = []
    param_group += [{'params': net.brnet.parameters(), 'lr': lr}]
    param_group += [{'params': net.extras.parameters(), 'lr': lr}]
    param_group += [{'params': net.fpn_topdown.parameters(), 'lr': lr}]
    param_group += [{'params': net.fpn_latlayer.parameters(), 'lr': lr}]
    param_group += [{'params': net.fpn_fem.parameters(), 'lr': lr}]
    param_group += [{'params': net.loc_pal1.parameters(), 'lr': lr}]
    param_group += [{'params': net.conf_pal1.parameters(), 'lr': lr}]
    param_group += [{'params': net.loc_pal2.parameters(), 'lr': lr}]
    param_group += [{'params': net.conf_pal2.parameters(), 'lr': lr}]
    param_group += [{'params': net.brnet.ref.parameters(), 'lr': lr / 10.}]
    param_group += [{'params': net.brnet.dark.parameters(), 'lr': lr / 10.}]

    # change SGD to AdamW
    optimizer = optim.Adamw(param_group, lr=lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)


    if args.cuda:
        if args.multigpu:
            net = net.to(device)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], find_unused_parameters=True)

            net_enh = net_enh.to(device)
            net_enh = torch.nn.parallel.DistributedDataParallel(net_enh, device_ids=[local_rank])

        
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

    if args.multigpu: # process group destroy
        dist.destroy_process_group()


    
    


if __name__ == '__main__':
    train()
