# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import transformers
from transformers import TrainingArguments


from arguments import argument_parser
from models.data.config import cfg
from models.data.widerface import WIDERDetection, detection_collate
from models.factory import build_net
from models.modules.enhancer import RetinexNet
from BRTrainer_v2 import BR_Trainer_v2 as BRTrainer


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    # Set random seed for reproducibility
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize WandB
    if args.use_wandb:
        wandb.init(project="DSFD", name="BRNet_Training", config=args)
        wandb.config.update(args)

    # data loader
    train_dataset = WIDERDetection(cfg.FACE.TRAIN_FILE, mode='train')
    val_dataset = WIDERDetection(cfg.FACE.VAL_FILE, mode='val')
    data_collator = detection_collate

    # model
    net = build_net("train", cfg.NUM_CLASSES)
    net_enh = RetinexNet()
    net_enh.load_state_dict(torch.load(args.save_folder + 'decomp.pth'))
    net_enh.to(device)


    net.extras.apply(net.weights_init)
    net.fpn_topdown.apply(net.weights_init)
    net.fpn_latlayer.apply(net.weights_init)
    net.fpn_fem.apply(net.weights_init)
    net.loc_pal1.apply(net.weights_init)
    net.conf_pal1.apply(net.weights_init)
    net.loc_pal2.apply(net.weights_init)
    net.conf_pal2.apply(net.weights_init)
    net.brnet.apply(net.weights_init)

    lr = args.lr * np.round(np.sqrt(args.batch_size / 4 * torch.cuda.device_count()),4)
    param_group = []
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
    param_group += [{'params': net.brnet.layers.parameters(), 'lr': lr}]

    optimizer = optim.SGD(param_group, lr=lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    training_args = TrainingArguments(
        output_dir=args.save_folder,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=lr,
        weight_decay=args.weight_decay,
        logging_dir=args.save_folder + 'logs',
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )

    trainer = BRTrainer(
        model=net,
        net_enh=net_enh,
        custom_optimizer = optimizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator = data_collator,
        use_wandb=args.use_wandb
    )

    trainer.train()
    trainer.save_model(os.path.join(args.save_folder, "best_model"))


    

if __name__ == '__main__':
    arguments = argument_parser()
    main(arguments)
