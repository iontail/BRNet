import torch
import torch.nn as nn

from .layers.basic_layers import *
from .layers.PhotoReceptor_module import Photoreceptor_Block
from .layers.DSFD_basic_modules import Interpolate
from .data.config import cfg

class BRNet(nn.Module):
    """
    Bio-reflect Net work (BRNet) -> Bio-Receptor Network??
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
        self.layer_num_cfg = [64, 64, 'M', 128, 128, 
                              'M', 256, 256, 
                              'C', 512, 512, 
                              'M', 512, 512, 
                              'M'] # + Conv7 + ReLU /+ Conv8 + ReLU

        """
        Check later index:
        [0, 'Conv2d'] [1, 'BatchNorm2d'] [2, 'ReLU']
        [3, 'Conv2d'] [4, 'BatchNorm2d'] [5, 'ReLU'] [6, 'MaxPool2d']

        [7, 'Conv2d'] [8, 'BatchNorm2d'] [9, 'ReLU']
        [10, 'Conv2d'] [11, 'BatchNorm2d'] [12, 'ReLU'] [13, 'MaxPool2d']
        [14, 'Photoreceptor_Block'] [15, 'Photoreceptor_Block'] 
        
        [16, 'MaxPool2d'] [17, 'Photoreceptor_Block'] [18, 'Photoreceptor_Block']
        
        [19, 'MaxPool2d'] [20, 'Photoreceptor_Block'] [21, 'Photoreceptor_Block']
        
        [22, 'MaxPool2d'] [23, 'Conv2d'] [24, 'ReLU'] [25, 'Conv2d'] [26, 'ReLU']
        """
        # index of each stage's layer
        self.stage_id = [[7, 8, 9, 10, 11, 12, 13, 14, 15],
                         [16, 17, 18],
                         [19, 20, 21],
                         [22, 23, 24, 25]] 
        self.cfg = cfg
        

        self.layers = []
        for i, v in enumerate(self.layer_num_cfg):
            if v == 'M':
                self.layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                self.layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                # reflectance 분기이후 한 스테이지는 VGG 그대로 진행
                # 나머지부터는 Photoreceptor block 적용
                if i < 6:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)

                    if cfg.NORMALIZE is not None: # if use normalization
                        self.layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    else:
                        self.layers += [conv2d, nn.ReLU(inplace=True)]

                else: # reflectance 이후에는 PhotoReceptor block 적용
                    self.layers += [Photoreceptor_Block(in_channels, v, normalize = cfg.NORMALIZE,
                                                        activation =  cfg.ACTIVATION,
                                                        reduction = cfg.REDUCTION,
                                                        cfg = cfg)]
                in_channels = v

        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.layers += [conv6,
                nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
        
        # for calculating reflectance   
        self.ref = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                Interpolate(2),
                nn.Conv2d(64, 3, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        
        self.dark = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                Interpolate(2),
                nn.Conv2d(64, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        
        self.layers = nn.ModuleList(self.layers)

    


    def forward(self, x, only_disentangle = False, hook_dict = None):
        """
        Args:
            x : input image (B, C, H, W)
            only_disentangle : whether to return reflectance and dark level only or not (default: False)
            
        Returns:
            sources         : list of feature maps from each stage
            darklevel_mean  : mean dark level (B, 1, 1, 1)
            ref             : reflectance map whose H and H equal to origial image (B, H, W, C) 
            ref_mean        : mean reflectance (B, 1, 1, 1)
        """
        sources = []

        
        for i in range(7):
            x = self.layers[i](x)
        sources += [x]

        if hook_dict is not None:
            if 'dark' in hook_dict and not only_disentangle:
                x.register_hook(hook_dict['dark'])
            if 'light' in hook_dict and only_disentangle:
                x.register_hook(hook_dict['light'])

        
        darklevel = self.dark(x) # (B, 1, H, W)
        darklevel_mean = darklevel.mean(dim = [2, 3], keepdim = True) # (B, 1, 1, 1)
        #print(f"DarkLevel: {darklevel.shape}")
        #print(f"DarkLevel_mean: {darklevel_mean.shape}")

        ref = self.ref(x) # (B, 3, H, W)
        ref_mean = ref.flatten(start_dim=2).mean(dim=-1) # (B, 3)
        ref_mean_average = ref_mean.mean(dim = 1, keepdim = True).reshape(-1, 1, 1, 1) # (B, 1, 1, 1)

        # reflectance만 알고싶을 때는, 끝까지 모델에 통과시킬 필요가 없음 
        if not only_disentangle:
            for id in self.stage_id:
                for i in id:
                    if self.layers[i].__class__.__name__ == 'Photoreceptor_Block':
                        x = self.layers[i](x, darkness_level = darklevel_mean.detach(), reflectance = ref_mean_average.detach()) #problem: detach?
                        # use detach to make darklevel/reflectance branch be as independent as possible 
                        # with photo receptor block
                    else:
                        x = self.layers[i](x)

                sources.append(x)

        return sources, darklevel_mean, ref, ref_mean
        
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
        sources, darklevel_mean, ref, ref_mean = detector(img, only_disentangle = False)

    for i, x in enumerate(sources):
        print(f'output of {i}th stage:', x.shape)
    print(f"reflectance map: {ref.shape}")
    print(f"reflectance_mean map: {ref_mean.shape}")
    print(f"darklevel_mean: {darklevel_mean.shape}")

    print("="*40)
    print("Check later index:")
    # Call _check_layer_idx directly on the detector instance
    detector._check_layer_idx()
    print("="*40)