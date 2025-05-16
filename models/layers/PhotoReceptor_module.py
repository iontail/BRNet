import torch
import torch.nn as nn
from layers.ops_dcnv3 import modules as opsm

str_to_act = {'relu': nn.ReLU(), 'gelu': nn.GELU()} # add the activation what you want to use
str_to_norm = {'bn': nn.BatchNorm2d, 'ln': nn.LayerNorm}

class GainBlock(nn.Module):
    """
    Strengthen the input signal 
    """
    def __init__(self, in_dim, h_dim, activation = 'relu'):
        super(GainBlock, self).__init__()

        self.in_dim = in_dim
        self.h_dim = h_dim
        self.activation = str_to_act[activation]

        # problems: 왜 중간에 차원을 4로 줄이지?
        # to reduce the number of parameters
        # acts like FFN but for efficiency 
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.in_dim, self.h_dim // 4, 1),
            self.activation,
            nn.Conv2d(self.h_dim // 4, self.h_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        gain = self.layers(x)
        return x * gain  + x
    
class ReflectiveFeedback(nn.Module):
    """
    Getting the reflectance information
    and add the reflactance information to each channel.

    act as **Tapetum Lucidum**
    """
    def __init__(self, h_dim, activation = 'relu'):
        super(ReflectiveFeedback, self).__init__()
        self.h_dim = h_dim
        self.activation = str_to_act[activation]
        self.layers = nn.Sequential(
            nn.Conv2d(1, h_dim, kernel_size = 1),
            self.activation
        )

    def forward(self, x, reflectance = None):
        if reflectance is not None:
            reflect_info = self.layers(reflectance)
            return x + reflect_info
        else:
            return x
        


# -----------------------------------------------------
# 1. Rod Path: 극저조도 노이즈 억제 + 에지 강조
# -----------------------------------------------------
class Rod_Block(nn.Module):
    """
    RodPath:

    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 reduction=2,
                 activation = 'relu',
                 normalize = 'bn',
                 downsample = False,
                 down = 1,
                 cfg = None):
        
        super(Rod_Block, self).__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.activation = activation
        self.normalize = normalize
        self.downsample = downsample
        self.cfg = cfg

        self.bn1 = str_to_norm[normalize](in_dim)
        self.bn2 = str_to_norm[normalize](out_dim)


        self.gain = GainBlock(in_dim, in_dim, activation = self.activation)
        self.tapetum = ReflectiveFeedback(in_dim, activation = self.activation) # problems: mid_dim??


        """
        We use the DCNv3 of InterImage by simply importing the ops_dcnv3 file
        Original code url:
            https://github.com/OpenGVLab/InternImage/blob/master/detection/mmdet_custom/models/backbones/intern_image.py#L377

        See https://arxiv.org/abs/2211.05778 for more details
        """
        self.dcn = opsm.DCNv3_pytorch(in_dim, kernel_size = 3, pad = 1, stride = 1)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size = 1, padding = 0) 

        self.layers = nn.Sequential(
            self.bn1,
            str_to_act[self.activation],
            self.conv,
            self.bn2,
            str_to_act[self.activation]
        )

        # not use spatial gating
        # becasue dcn acts as gating by capturing high-frequency informations
        

    def forward(self, x, reflectance = None):

        if self.cfg.ABLATION.USE_GAIN:
            x = self.gain(x)

        if self.cfg.ABLATION.USE_TAPETUM and reflectance is not None:
            x = self.tapetum(x, reflectance)

        # Permute to channels_last for DCNv3_pytorch
        x_permuted = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)

        # Apply DCNv3
        dcn_out = self.dcn(x_permuted) # Output is (N, H, W, C)

        # Permute back to channels_first for subsequent layers
        dcn_out = dcn_out.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        # Apply the rest of the layers (BN -> Act -> Conv -> BN -> Act)
        out = self.layers(dcn_out)
        return out

        
        out = self.layers(x)

        return out


# -----------------------------------------------------
# 2. Cone Path: 일반 조도, 컬러/디테일 유지
# -----------------------------------------------------
class Cone_Block(nn.Module):
    """
    ConePath:
      - 3×3 Conv 2개
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 normalize = 'bn',
                 activation = 'relu',
                 downsample = False,
                 down = 1
                 ):
        super(Cone_Block, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.normalize = normalize
        self.downsample = downsample
        self.activation = activation
        self.down = down

        self.bn1= str_to_norm[normalize](out_dim)
        self.bn2= str_to_norm[normalize](out_dim)
        self.activation = str_to_act[activation]

        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size = 1, padding = 0),
            self.bn1,
            self.activation,
            nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
            self.bn2,
            self.activation
        )


    def forward(self, x):
        out = self.layers(x)

        return out


# -----------------------------------------------------
# 3. PhotoreceptorModule: darkness level로 Rod/Cone 가중합
# -----------------------------------------------------
class Photoreceptor_Block(nn.Module):
    """
    PhotoreceptorModule:
      - RodPath & ConePath 병렬 처리
      - darkness_level(0~1)에 따라 가중합

      args:
        in_dim       : input channel
        out_dim      : output channel
        normalize    : normalization layer (bn, ln)
        activation    : activation function (relu, gelu)
        downsample   : downsample or not
        reduction    : reduction ratio for mid_dim
        down         : downsample ratio for the input feature map
        downsample   : downsample or not
        down         : downsample ratio for the input feature map
        cfg          : configuration

    returns:
        out          : [B, C, H, W] feature map
    """
    def __init__(
            self,
            in_dim,
            out_dim,
            normalize = 'bn',
            activation = 'relu',
            downsample = False,
            reduction = 2,
            down = 1,
            cfg = None
            ):
        super(Photoreceptor_Block, self).__init__()


        self.in_dim = in_dim
        self.out_dim = out_dim
        self.normalize = normalize
        self.activation = activation
        self.downsample = downsample
        self.down = down

        self.rod = Rod_Block(in_dim, out_dim, reduction, activation,
                             normalize, downsample, down, cfg = cfg)
        self.cone = Cone_Block(in_dim, out_dim, normalize, activation,
                               downsample, down)


    def forward(self, x, darkness_level, reflectance = None):
        """
        x              : [B, C, H, W] feature map from backbone
        darkness_level : [B, 1, 1, 1] scalar between [0,1], 1=밝음, 0=매우 어두움
        reflectance    : [B, 1, 1, 1] scalar
        """

        if reflectance is not None:
            assert reflectance.dim() == 4, "Reflectance has wrong shape. check whether the shape is [B, 1, 1, 1]"

            rod_out = self.rod(x, reflectance)
            cone_out = self.cone(x)

        else:
            rod_out = self.rod(x)
            cone_out = self.cone(x)
    


        # -------------------------------
        # Weighted sum
        # 어두우면 rod path 가중치 증가
        # 밝으면 cone path 가중치 증가
        # -------------------------------
        # 각 feature (rod_out & cone_out)은 서로 다른 도메인 feature여서 단순 addition은 최적의 융합이 아닐 수 있음
        # 그래서 concat 후에 추가적은 fc를 통해 적용해줄 수도 있음
        # 하지만, 본 모델은 detection head에 추가로 붙어서 보조적인 역할을 하기 때문에, 추가적인 연산을 최대한 자제해야 하고,
        # PhotoreceptorModule 아이디어 자체가 “조도별 gating” 취지이므로, Weighted sum으로 결정
        # -------------------------------
        out = (1- darkness_level) * rod_out + darkness_level * cone_out
        return out

#------------------
# Test
#------------------
if __name__ == '__main__':
    # mac 환경이라서 그냥 cpu
    device = torch.device('cpu')
    batch = 2
    res = 512

    from easydict import EasyDict
    cfg = EasyDict()
    cfg.ABLATION = EasyDict()
    cfg.ABLATION.DARKLEVEL = True 
    cfg.ABLATION.REFLECTANCE = True
    cfg.ABLATION.ILLUMINATION = True

    cfg.ABLATION.SOTR = True
    cfg.ABLATION.USE_GAIN = True
    cfg.ABLATION.USE_TAPETUM = True

    detector = Photoreceptor_Block(4, 10, normalize = 'bn', activation = 'relu', downsample = False, down = 1, cfg = cfg).to(device)
    
    img = torch.randn(batch, 4, res, res).to(device)
    darkness_level = torch.randn(batch, 1, 1, 1).to(device)
    
    detector.eval()
    

    def count(block):
        return sum(p.numel() for p in block.parameters()) / 10 ** 6
    print(f'Params: {count(detector)} milions')


    with torch.no_grad():
        output = detector(img, darkness_level)

    for i, x in enumerate(output):
        print(f'output of {i}th stage:', x.shape)