import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from modules.basic_layers import *
from modules.PhotoReceptor_module import Photoreceptor_Block
from vgg import vgg


class Bio_ResNet(nn.Module):
    def __init__(self, layers):

        super(Bio_ResNet, self).__init__()

        self.layers = layers


    def forward(self, x, reflectance = False):
        pass


class Bio_VGGNet(nn.Module):
    def __init__(self, batch_norm = False, in_channels = 3):

        super(Bio_VGGNet, self).__init__()
        self.config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512, 'M']

        self.stage_id = [[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21], [22, 23], [24, 25]] # index of each stage's layer

        self.layers = []
        for i, v in enumerate(self.config):
            if v == 'M':
                self.layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                self.layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                # replace vanilla conv2d with PhotoReceptor block
                # 기존에는 stem cell 이후에 reflectance 분기했지만, 우리는 조금 더 가서 1st stage에서 분기
                if i < 6:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)

                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    else:
                        self.layers += [conv2d, nn.ReLU(inplace=True)]

                else: # reflectance 이후에는 PhotoReceptor block 적용
                    if batch_norm:
                        self.layers += [Photoreceptor_Block(in_channels, v, normalize = 'bn')]
                    else:
                        self.layers += [Photoreceptor_Block(in_channels, v)]

                in_channels = v

        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.layers += [conv6,
                nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
        
        # for calculating reflectance
        self.fc = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )


    def forward(self, x, reflectance = False):
        # reflectance만 알고싶을 때는, 끝까지 모델에 통과시킬 필요가 없음
        if reflectance:
            for i in range(10):
                x = self.layers[i](x)

            # return reflectance only
            return [x]
        
        else:
            sources = []
            for i in range(10):
                x = self.layers[i](x)

            B, C, H, W = x.size()
            darklevel = self.fc(x.reshape(B, H, W, C)) # (B, H, W, 1)
            darklevel = darklevel.mean(dim = [1, 2], keepdim = True) # (B, 1, 1, 1)
            sources.append(x)
                

            for id in self.stage_id:
                for i in id:
                    if self.layers[i].__class__.__name__ == 'Photoreceptor_Block':
                        x = self.layers[i](x, darkness_level = darklevel)
                    else:
                        x = self.layers[i](x)

                sources.append(x)

            return sources
        
    def _check_layer_idx(self):
        for i, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__
            print(f'{[i, layer_name]}')

        





class Bio_ReflectNet(nn.Module):
    """
    Args:
    backbone -> feature extractor (one of vgg and resnet)
    """
    def __init__(self, backbone):
        
        super(Bio_ReflectNet, self).__init__()

        self.backbone = backbone
        self.channels = 64
        

        # make model by chaning 'backbone model' 'PhotoReceptor_module based model'

        assert backbone in ['vgg', 'resnet50', 'resnet101', 'resnet152'], \
            "Invalid backbone model. Please set the argument between ['vgg', 'resnet50', 'resnet101', 'resnet152']"

        if self.backbone == 'vgg':
            self.model = Bio_VGGNet()
            
        elif self.backbone == 'resnet50':
            layers = [3, 4, 6, 3]
            self.model = self._make_resnet_model(layers)

        elif self.backbone == 'resnet101':
            layers = [3, 4, 23, 3]
            self.model = self._make_resnet_model(layers)
        
        elif self.backbone == 'resnet152':
            layers = [3, 8, 36, 3]
            self.model = self._make_resnet_model(layers)

    def _make_resnet_model(self, layers):
        pass

    def forward(self, x, reflectance = False):
        
        # ----------------------------------------------------------------
        # 원래는 Reflectance mapping은 stem cell 이후인데, 우리는 conv1 이후에 적용
        # ----------------------------------------------------------------

        # reflectance만 반환할 떄는, DSFD에서 isinstance를 사용해서 
        # list인지 구별후 인덱싱하여 사용
        if 'resnet' in self.backbone:
            sources = self.model(x, reflectance = reflectance)
            
        else:
            sources = self.model(x, reflectance = reflectance)
            

        return sources
    

if __name__ == '__main__':
    # mac 환경이라서 그냥 cpu
    device = torch.device('cpu')
    batch = 3
    res = 512
    detector = Bio_ReflectNet(backbone = 'vgg').to(device)
    
    img = torch.randn(batch, 3, res, res).to(device)

    detector.eval()

    
    def count(block):
        return sum(p.numel() for p in block.parameters()) / 10 ** 6
    print(f'Params: {count(detector)} milions')


    with torch.no_grad():
        output = detector(img, reflectance = False)

    for i, x in enumerate(output):
        print(f'output of {i}th stage:', x.shape)