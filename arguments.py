import argparse

def config():
    parser = argparse.ArgumentParser(description='DSFD face Detector Training With Pytorch')

    parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
    parser.add_argument('--model', default='dark', type=str, choices=['dark', 'vgg', 'resnet50', 'resnet101', 'resnet152'], help='Model for training')
    parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')
    parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float, help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--multigpu', default=False, type=bool, help='Use multiple GPU training')
    parser.add_argument('--save_folder', default='weights/', help='Directory for saving checkpoint models')
    parser.add_argument('--local_rank', type=int, help='Local rank for distributed training')
    parser.add_argument('--use_wandb', default=True, type=bool, help='Whether using wandb log')

    args = parser.parse_args()
    
    return args