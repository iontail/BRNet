import os
import warnings
warnings.filterwarnings("ignore")

import torch
import random
import numpy as np
import wandb
import torch
import pandas as pd
from tqdm import tqdm

from safetensors.torch import load_model

from arguments import get_arguments

from src.dataset.get_dataset import get_dataset
from src.model.get_model import get_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

def main(args):
    set_seed(args.seed)

    # Load dataset
    test_datset, _, data_collator = get_dataset(args)

    test_dataloader = torch.utils.data.DataLoader(test_datset, 
                                                batch_size=args.per_device_eval_batch_size, 
                                                collate_fn=data_collator,
                                                drop_last=False,
                                                shuffle=False)

    # Load Model 
    model = get_model(args)
    load_model(model, args.load_dir + '/model.safetensors')

    model.to(device)
    model.eval()

    prediciton = []

    with torch.no_grad():
        for features in tqdm(test_dataloader):
            inputs = features['inputs'].to(device)
            outputs = model(inputs)
            prediciton.append(outputs.cpu().numpy())

    prediciton = np.concatenate(prediciton, axis=0)
    
if __name__=="__main__":

    args = get_arguments()
    main(args)