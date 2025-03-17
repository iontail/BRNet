import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from modules.basic_layers import *
from modules.PhotoReceptor_module import Photoreceptor_Block, Photoreceptor_ResNet_Block
from resnet import Bottleneck


class Bio_ResNet(nn.Module):
    def __init__(self, layer_list, expansion = 4):

        super(Bio_ResNet, self).__init__()

        self.layer_list = layer_list
        self.expansion = expansion
        self.channels = 64

        self.layers = []
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(Bottleneck, 64, layer_list[0])
        self.layer2 = self._make_layer(Photoreceptor_ResNet_Block, 128, layer_list[1], stride=2)
        self.layer3 = self._make_layer(Photoreceptor_ResNet_Block, 256, layer_list[2], stride=2)
        self.layer4 = self._make_layer(Photoreceptor_ResNet_Block, 512, layer_list[3], stride=2)

        # for calculating reflectance
        self.fc = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
  

    def _make_layer(self, block, channels, blocks, stride=1):
        
        assert block.__name__ in ['Photoreceptor_ResNet_Block', 'Bottleneck'], \
            f"{block.__name__} is an invalid Block. Block name must be one of ['Photoreceptor_ResNet_Block', 'Bottleneck']"
        
        layers = []
        downsample = False
        if stride != 1 or self.channels != channels * block.expansion:
            downsample = True

        if block.__name__ == 'Photoreceptor_ResNet_Block':
            layers.append(block(self.channels, channels, stride = stride))
            
            self.channels  = channels * block.expansion
            for _ in range(1, blocks):  
                layers.append(block(self.channels, channels))
        else:
            layers.append(block(self.channels, channels, downsample = downsample, down = stride))

            self.channels  = channels * block.expansion
            for _ in range(1, blocks):  
                layers.append(block(self.channels, channels))
        
        return nn.Sequential(*layers)

    def forward(self, x, reflectance = False):
        sources = []
        # reflectance만 알고싶을 때는, 끝까지 모델에 통과시킬 필요가 없음
        for i in range(2):
            x = getattr(self, f"layer{i}")(x)
            sources += [x]

        B, C, H, W = x.size()
        darklevel = self.fc(x.reshape(B, H, W, C)) # (B, H, W, 1)
        darklevel = darklevel.mean(dim = [1, 2], keepdim = True) # (B, 1, 1, 1)

        if not reflectance:
            for i in range(2, 5):
                layer = getattr(self, f"layer{i}")
                for block in layer:
                    if isinstance(block, Photoreceptor_ResNet_Block):
                        x = block(x, darkness_level=darklevel)
                    else:
                        x = block(x)
                sources += [x]


        return sources, darklevel



class Bio_VGGNet(nn.Module):
    def __init__(self, batch_norm = False, in_channels = 3):

        super(Bio_VGGNet, self).__init__()
        #self.config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512, 'M']
        # Bio-Receptor module은 개당 가지고 있는 conv수가 많아서 config 변경
        self.config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'C', 512, 512, 'M', 512, 512, 'M']

        self.stage_id = [[6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16], [17, 18]] # index of each stage's layer

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
        
        self.layers = nn.ModuleList(self.layers)

    


    def forward(self, x, reflectance = False):
        sources = []

        
        for i in range(5):
            x = self.layers[i](x)
        sources += [x]

        for i in range(5, 10):
            x = self.layers[i](x)
        sources += [x]


        B, C, H, W = x.size()
        darklevel = self.fc(x.reshape(B, H, W, C)) # (B, H, W, 1)
        darklevel = darklevel.mean(dim = [1, 2], keepdim = True) # (B, 1, 1, 1)

        # reflectance만 알고싶을 때는, 끝까지 모델에 통과시킬 필요가 없음 
        if not reflectance:
            for id in self.stage_id:
                for i in id:
                    if self.layers[i].__class__.__name__ == 'Photoreceptor_Block':
                        x = self.layers[i](x, darkness_level = darklevel)
                    else:
                        x = self.layers[i](x)

                sources.append(x)
                print(x.shape)

        return sources, darklevel
        
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
            self.model = Bio_ResNet(layers)

        elif self.backbone == 'resnet101':
            layers = [3, 4, 23, 3]
            self.model = Bio_ResNet(layers)
        
        elif self.backbone == 'resnet152':
            layers = [3, 8, 36, 3]
            self.model = Bio_ResNet(layers)

    def forward(self, x, reflectance = False):
        
        # ----------------------------------------------------------------
        # 원래는 Reflectance mapping은 stem cell 이후인데, 우리는 conv1 이후에 적용
        # ----------------------------------------------------------------

        # reflectance만 반환할 떄는, DSFD에서 isinstance를 사용해서 
        # list인지 구별후 인덱싱하여 사용
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
        output, darklevel = detector(img, reflectance = False)

    for i, x in enumerate(output):
        print(f'output of {i}th stage:', x.shape)
    print(f"darklevel: {darklevel.shape}")

    print("="*40)
    print("Check later index:")
    detector.model._check_layer_idx()
    print("="*40)