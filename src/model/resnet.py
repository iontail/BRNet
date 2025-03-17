import torch
import torch.nn as nn
from modules.basic_layers import *

class Resnet_Basic_Block(nn.Module):
    def __init__(self,
                 in_channels,                    # Number of input channels.
                 out_channels,                   # Number of output channels.
                 kernel_size,                    # Convolution kernel size.
                 bias            = False,         # Use bias?
                 activation      = 'relu',       # Activation function name
                 downsample        = False,        # Use downsampling?
                 up              = 1,            # Integer upsampling factor.
                 down            = 1,            # Integer downsampling factor.
                 groups          = 1,            # Number of groups for Grouped convolution.
                 ):
        
        super(Resnet_Basic_Block, self).__init__()
        self.up = up
        self.down = down
        self.downsample = downsample


        if downsample:
            self.downsample = Conv2dLayer(in_channels, out_channels, kernel_size = 1, normalize = 'bn', up = up, down = down)

        self.block1 = Conv2dLayer(in_channels, out_channels, kernel_size, bias = bias, activation = 'relu',
                                  normalize = 'bn', up = up, down = down, groups = groups)
        self.block2 = Conv2dLayer(out_channels, out_channels, kernel_size, bias = bias, activation = 'linear',
                                  normalize = 'bn', up = up, down = down, groups = groups)
        self.relu = nn.ReLU()


    def forward(self, x):
        residual = x

        out = self.block1(x)
        out = self.block2(out)

        if self.downsample:
            residual = self.downsample(residual)

        out  += residual
        out = self.relu(out)
        return out
    

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 in_channels,                    # Number of input channels.
                 out_channels,                   # Number of output channels.
                 bias            = False,        # Use bias?
                 downsample      = False,        # Use downsampling?
                 up              = 1,            # Integer upsampling factor.
                 down            = 1,            # Integer downsampling factor.
                 groups          = 1,            # Number of groups for Grouped convolution.
                 ):
        
        super(Bottleneck, self).__init__()

        # stride는 down을 통해서 control
        self.up = up
        self.down = down
        self.downsample = downsample



        self.block1 = Conv2dLayer(in_channels, out_channels, kernel_size = 1, bias = bias, 
                                  activation = 'relu', normalize = 'bn', down = 1, groups = groups)
        self.block2 = Conv2dLayer(out_channels, out_channels, kernel_size = 3, bias = bias, 
                                  activation = 'relu', normalize = 'bn', down = down, groups = groups)
        self.block3 = Conv2dLayer(out_channels, out_channels * self.expansion, kernel_size = 1, bias = bias, 
                                  activation = 'linear', normalize = 'bn', down = 1, groups = groups)

        if self.downsample:
            self.downsampling = Conv2dLayer(in_channels, out_channels * self.expansion, kernel_size = 1, normalize = 'bn', down = down)

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)

        if self.downsample:
            residual = self.downsampling(residual)

        out += residual
        out = self.relu(out)

        return out
    
#------------------
# ResNet class
#------------------
class ResNet(nn.Module):
    def __init__(self, block, layers):

        super(ResNet, self).__init__()
        self.channels = 64

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, channels, blocks, stride=1):
        layers = []
        downsample = False
        if stride != 1 or self.channels != channels * block.expansion:
            downsample = True

        layers.append(block(self.channels, channels, downsample = downsample, down = stride))
        self.channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        sources = []

        x = self.layer0(x)

        # save multi-scale features
        for i in range(1, 5):
            x = getattr(self, f"layer{i}")(x)
            sources += [x]

        return sources



def resnet50():
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model


def resnet101():
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    return model


def resnet152():
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    return model


#------------------
# Test
#------------------
if __name__ == '__main__':
    # mac 환경이라서 그냥 cpu
    device = torch.device('cpu')
    batch = 2
    res = 512
    detector = resnet50().to(device)
    
    img = torch.randn(batch, 3, res, res).to(device)

    detector.eval()

    
    def count(block):
        return sum(p.numel() for p in block.parameters()) / 10 ** 6
    print(f'Params: {count(detector)} milions')


    with torch.no_grad():
        output = detector(img)

    for i, x in enumerate(output):
        print(f'output of {i}th stage:', x.shape)