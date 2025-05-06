import torch
import torch.nn as nn
from bioreflectnet.models.layers.basic_layers import *

vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
           512, 512, 512, 'M']

def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return nn.ModuleList(layers)


#------------------
# Test
#------------------
if __name__ == '__main__':
    # mac 환경이라서 그냥 cpu
    device = torch.device('cpu')
    batch = 2
    res = 512
    detector = vgg(vgg_cfg, 3).to(device)
    
    img = torch.randn(batch, 3, res, res).to(device)
    
    detector.eval()


    def count(block):
        return sum(p.numel() for p in block.parameters()) / 10 ** 6
    print(f'Params: {count(detector)} milions')


    with torch.no_grad():
        for i, layer in enumerate(detector):
            img = layer(img)
            layer_name = layer.__class__.__name__
            print(f"Output of {i}th layer ({layer_name}): {img.shape}")



"""
Name of each k-th layer:

[['0th', 'Conv2d'], ['1th', 'ReLU'], ['2th', 'Conv2d'],
['3th', 'ReLU'], ['4th', 'MaxPool2d'], ['5th', 'Conv2d'],
['6th', 'ReLU'], ['7th', 'Conv2d'], ['8th', 'ReLU'],
['9th', 'MaxPool2d'], ['10th', 'Conv2d'], ['11th', 'ReLU'],
['12th', 'Conv2d'], ['13th', 'ReLU'], ['14th', 'Conv2d'],
['15th', 'ReLU'], ['16th', 'MaxPool2d'], ['17th', 'Conv2d'],
['18th', 'ReLU'], ['19th', 'Conv2d'], ['20th', 'ReLU'],
['21th', 'Conv2d'], ['22th', 'ReLU'], ['23th', 'MaxPool2d'],
['24th', 'Conv2d'], ['25th', 'ReLU'], ['26th', 'Conv2d'], ['27th', 'ReLU'],
['28th', 'Conv2d'], ['29th', 'ReLU'], ['30th', 'MaxPool2d'], ['31th', 'Conv2d'],
['32th', 'ReLU'], ['33th', 'Conv2d'], ['34th', 'ReLU']]
"""


    