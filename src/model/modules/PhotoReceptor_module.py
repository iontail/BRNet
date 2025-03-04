import torch
import torch.nn as nn
from modules.basic_layers import *


#------------------
# Rode Block과 Cone Block은 프로토타임임
#------------------

# -----------------------------------------------------
# 1. Rod Path: 극저조도 노이즈 억제 + 에지 강조
# -----------------------------------------------------
class Rod_Block(nn.Module):
    """
    RodPath:
      - 1×1 Conv로 채널 축소
      - 3×3 Conv 2회로 노이즈 억제 및 에지 강조
    """
    def __init__(self, in_channels, out_channels, reduction=2, normalize = 'bn', downsample = False, down = 1):
        super(Rod_Block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.downsample = downsample


        mid_channels = in_channels // reduction

        #----------------------------------
        # GroupNorm도 고려해볼것 (배치 너무 작으면)
        #----------------------------------

        self.conv1 = Conv2dLayer(in_channels, mid_channels, kernel_size=1, activation = 'relu', normalize = normalize)
        #self.norm1 = nn.GroupNorm(num_groups=4, num_channels=mid_channels)
        self.conv2 = Conv2dLayer(mid_channels, mid_channels, kernel_size=3, activation = 'relu', normalize = normalize, down = down)
        self.conv3 = Conv2dLayer(mid_channels, out_channels, kernel_size=3, activation = 'linear', normalize = normalize)




        
        if (self.downsample) or (in_channels != out_channels):
            self.downsampling = Conv2dLayer(in_channels, out_channels, kernel_size=1,
                                            activation = 'linear', normalize = normalize, down = down, bias = False)

 
        self.act = nn.ReLU(inplace=True)

    

    def forward(self, x):

        # rod path는 "원본 + 노이즈"를 처리
        # normal-lit 상태도 중요하지만, low-light 처리가 우선적임
        # residual connection의 개수는 최소화
        # 해당 부분의 내용은 더 깊이 고민해서 논문에 반영
        residual = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        
        if self.downsample or(self.in_channels != self.out_channels):
            residual = self.downsampling(residual)

        out += residual
        out = self.act(out)

        return out


# -----------------------------------------------------
# 2. Cone Path: 일반 조도, 컬러/디테일 유지
# -----------------------------------------------------
class Cone_Block(nn.Module):
    """
    ConePath:
      - 3×3 Conv 2회
      - 기본 BatchNorm (or GroupNorm) + ReLU
      - 너무 많은 필터링 없이 원본 디테일 보존
    """
    def __init__(self, in_channels, out_channels, normalize = 'bn', downsample = False, down = 1):
        super(Cone_Block, self).__init__()

        self.downsample = downsample


        # 레이어 2개 밖에 없어서 downsampling하게 되면, 첫 번쨰 레이어에 down 적용
        self.conv1 = Conv2dLayer(in_channels, out_channels, kernel_size=3, activation = 'relu', normalize = normalize, down = down)
        self.conv2 = Conv2dLayer(out_channels, out_channels, kernel_size=3, activation = 'relu', normalize = normalize)



    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out


# -----------------------------------------------------
# 3. PhotoreceptorModule: darkness level로 Rod/Cone 가중합
# -----------------------------------------------------
class Photoreceptor_Block(nn.Module):
    """
    PhotoreceptorModule:
      - RodPath & ConePath 병렬 처리
      - darkness_level(0~1)에 따라 가중합
    """
    def __init__(self, in_channels, out_channels, reduction=2, stride = 1):
        super(Photoreceptor_Block, self).__init__()

        if stride != 1:
            self.downsample = True
        else:
            self.downsample = False

        self.rod_path = Rod_Block(in_channels, out_channels, reduction, downsample = self.downsample, down = stride)
        self.cone_path = Cone_Block(in_channels, out_channels, downsample = self.downsample, down = stride)



    def forward(self, x, darkness_level):
        """
        x              : [B, C, H, W] feature map from backbone
        darkness_level : [B, 1, 1, 1], scalar in [0,1], 1=밝음, 0=매우 어두움
        """

        rod_out = self.rod_path(x)
        cone_out = self.cone_path(x)



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


class Rod_ResNet_Block(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, reduction=2, normalize = 'bn', downsample = False, down = 1):
        super(Rod_ResNet_Block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.downsample = downsample



        self.conv1 = Conv2dLayer(in_channels, out_channels, kernel_size=1, activation = 'relu', normalize = normalize)
        #self.norm1 = nn.GroupNorm(num_groups=4, num_channels=mid_channels)
        self.conv2 = Conv2dLayer(out_channels, out_channels, kernel_size=3, activation = 'relu', normalize = normalize, down = down)
        self.conv3 = Conv2dLayer(out_channels, out_channels * self.expansion, kernel_size=3, activation = 'linear', normalize = normalize)




        
        if (self.downsample) or (in_channels != out_channels * self.expansion):
            self.downsampling = Conv2dLayer(in_channels, out_channels * self.expansion, kernel_size=1,
                                            activation = 'linear', normalize = normalize, down = down, bias = False)

 
        self.act = nn.ReLU(inplace=True)


    def forward(self, x):

        # rod path는 "원본 + 노이즈"를 처리
        # normal-lit 상태도 중요하지만, low-light 처리가 우선적임
        # residual connection의 개수는 최소화
        # 해당 부분의 내용은 더 깊이 고민해서 논문에 반영
        residual = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        
        if self.downsample or(self.in_channels != self.out_channels * self.expansion):
            residual = self.downsampling(residual)

        out += residual
        out = self.act(out)

        return out


class Cone_ResNet_Block(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, normalize = 'bn', downsample = False, down = 1):
        super(Cone_ResNet_Block, self).__init__()

        self.downsample = downsample


        # 레이어 2개 밖에 없어서 downsampling하게 되면, 첫 번쨰 레이어에 down 적용
        self.conv1 = Conv2dLayer(in_channels, out_channels, kernel_size=3, activation = 'relu', normalize = normalize, down = down)
        self.conv2 = Conv2dLayer(out_channels, out_channels * self.expansion, kernel_size=3, activation = 'relu', normalize = normalize)



    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out


class Photoreceptor_ResNet_Block(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, reduction=2, stride = 1):
        super(Photoreceptor_ResNet_Block, self).__init__()

        if stride != 1:
            self.downsample = True
        else:
            self.downsample = False

        self.rod_path = Rod_ResNet_Block(in_channels, out_channels, reduction, downsample = self.downsample, down = stride)
        self.cone_path = Cone_ResNet_Block(in_channels, out_channels, downsample = self.downsample, down = stride)



    def forward(self, x, darkness_level):
        """
        x              : [B, C, H, W] feature map from backbone
        darkness_level : [B, 1, 1, 1], scalar in [0,1], 1=밝음, 0=매우 어두움
        """

        rod_out = self.rod_path(x)
        cone_out = self.cone_path(x)



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
    detector = Photoreceptor_ResNet_Block(3, 10, reduction = 2, stride = 1).to(device)
    
    img = torch.randn(batch, 3, res, res).to(device)
    darkness_level = torch.randn(batch, 1, 1, 1).to(device)
    
    detector.eval()
    

    def count(block):
        return sum(p.numel() for p in block.parameters()) / 10 ** 6
    print(f'Params: {count(detector)} milions')


    with torch.no_grad():
        output = detector(img, darkness_level)

    for i, x in enumerate(output):
        print(f'output of {i}th stage:', x.shape)