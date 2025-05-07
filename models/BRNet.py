import torch
import torch.nn as nn

from bioreflectnet.models.layers.basic_layers import *
from bioreflectnet.models.layers.PhotoReceptor_module import Photoreceptor_Block
from bioreflectnet.models.layers.DSFD_basic_modules import Interpolate
from bioreflectnet.data.config import cfg

class BRNet(nn.Module):
    """
    Bio-reflect Network (BRNet) -> Bio-Receptor Network??
    Modified VGGNet

    Args:
    
        batch_norm  :   whether to use batch normalization or not
        in_channels :   number of input channels (default: 3)
        cfg         :   configuration settings (default: cfg)
    """
    def __init__(self, in_channels = 3, cfg = cfg):

        super(BRNet, self).__init__()
        #self.config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512, 'M']
        # Bio-Receptor module은 개당 가지고 있는 conv수가 많아서 config 변경
        self.config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'C', 512, 512, 'M', 512, 512, 'M']

        self.stage_id = [[6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16], [17, 18]] # index of each stage's layer
        self.cfg = cfg
        

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

                    if cfg.NORMALIZE is not None: # if use normalization
                        self.layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    else:
                        self.layers += [conv2d, nn.ReLU(inplace=True)]

                else: # reflectance 이후에는 PhotoReceptor block 적용
                    self.layers += [Photoreceptor_Block(in_channels, v, normalize = cfg.NORMALIZE,
                                                        activation =  cfg.ACTIVATION,
                                                        reduction = cfg.REDUCTION)]

        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.layers += [conv6,
                nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
        
        # for calculating reflectance   
        self.ref = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                Interpolate(2),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        
        self.illuminaion = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                Interpolate(2),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        
        self.dark = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                Interpolate(2),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        
        self.layers = nn.ModuleList(self.layers)

    


    def forward(self, x, only_ref = False):
        """
        Args:
            x : input image (B, C, H, W)
            only_ref : whether to return reflectance only or not (default: False)
            
        Returns:
            sources : list of feature maps from each stage
            ref     : reflectance map (B, 1, 1, 1)
            illi    : illumination map (B, 1, 1, 1)
            darklevel: darkness level (B, 1, 1, 1)    
        """
        sources = []

        
        for i in range(5):
            x = self.layers[i](x)
        sources += [x]

        for i in range(5, 10):
            x = self.layers[i](x)
        sources += [x]


        B, C, H, W = x.size()
        darklevel = self.dark(x.reshape(B, H, W, C)) # (B, H, W, 1)
        darklevel = darklevel.mean(dim = [1, 2], keepdim = True) # (B, 1, 1, 1)

        ref = self.ref(x.reshape(B, H, W, C))
        ref = ref.mean(dim = [1, 2], keepdim = True) # (B, 1, 1, 1)

        illi = self.illuminaion(x.reshape(B, H, W, C))
        illi = illi.mean(dim = [1, 2], keepdim = True) # (B, 1, 1, 1)

        # reflectance만 알고싶을 때는, 끝까지 모델에 통과시킬 필요가 없음 
        if not only_ref:
            for id in self.stage_id:
                for i in id:
                    if self.layers[i].__class__.__name__ == 'Photoreceptor_Block':
                        x = self.layers[i](x, darkness_level = darklevel, reflectance = ref)
                    else:
                        x = self.layers[i](x)

                sources.append(x)
                print(x.shape)

        return sources, ref, illi, darklevel
        
    def _check_layer_idx(self):
        for i, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__
            print(f'{[i, layer_name]}')



if __name__ == '__main__':
    # mac 환경이라서 그냥 cpu
    device = torch.device('cpu')
    batch = 3
    res = 512
    # Instantiate BRNet correctly, without the 'backbone' argument
    detector = BRNet(cfg=cfg).to(device)
    
    img = torch.randn(batch, 3, res, res).to(device)

    detector.eval()

    def count(block):
        return sum(p.numel() for p in block.parameters()) / 10 ** 6
    print(f'Params: {count(detector)} milions')


    with torch.no_grad():
        # Unpack all returned values from the forward method
        sources, ref, illi, darklevel = detector(img, only_ref = False)

    for i, x in enumerate(sources):
        print(f'output of {i}th stage:', x.shape)
    print(f"reflectance map: {ref.shape}")
    print(f"illumination map: {illi.shape}")
    print(f"darklevel: {darklevel.shape}")

    print("="*40)
    print("Check later index:")
    # Call _check_layer_idx directly on the detector instance
    detector._check_layer_idx()
    print("="*40)