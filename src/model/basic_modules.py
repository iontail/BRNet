import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from src.model.deformable_conv_v3 import Deformable_Conv2d


#--------------------- 모듈 만드는 데에 필요한 함수 정의 -----------------------
def get_style_code(a, b): # 그냥 concat
    return torch.cat([a, b], dim=1)


def token2feature(x, x_size):
    B, N, C = x.shape
    h, w = x_size
    x = x.permute(0, 2, 1).reshape(B, C, h, w)
    return x


def feature2token(x): 
    B, C, H, W = x.shape
    x = x.view(B, C, -1).transpose(1, 2)
    return x

def nf(stage, channel_base=32768, channel_decay=1.0, channel_max=512):
    NF = {512: 64, 256: 128, 128: 256, 64: 512, 32: 512, 16: 512, 8: 512, 4: 512}
    return NF[2 ** stage]

class linear(nn.Module):
    def __init__(self):
        super(linear, self).__init__()
    
    def forward(self, x):
        return x

def get_activation(activation, params: dict = None):
    for i in activation:
        # 소문자로만 activation func 받을 것임
        assert (97 <= ord(i) <= 122), f'Acitvation should be lower. Check the argument: {activation}'

    # [activation, gain]
    act_dict = {'relu': [nn.ReLU, np.sqrt(2)], 'gelu': [nn.GELU, np.sqrt(2)], 'selu': [nn.SELU, 3/4], 'sigmoid': [nn.Sigmoid, 1],
                'softplus': [nn.Softplus, 1], 'lrelu': [nn.LeakyReLU, np.sqrt(2)], 'linear': [linear, 1]}
    
    
    assert activation in act_dict, f"{activation} is not in our activation dictionary."

    return act_dict[activation]

def bias_act(x, b = None, act = 'linear', dim = None, gain = 1):

    if dim is not None and b is not None:
        b_shape = [1 for _ in range(len(x.shape))]
        b_shape[dim] = x.size(dim)

        b = b.reshape(*b_shape)

    if b is not None:
        x = x + b

    activation, act_gain = get_activation(act)
    out = activation()(x)
    #out = out * (gain * act_gain)
    #우리는 gain 고려 안 할 거임 (하면 좋긴하다)

    return out

def normalize_2nd_moment(x, dim=1, eps=1e-8):
    # x * sqrt(x^2.mean)
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

    
#------------------------------------------------------------
# 모듈 정의 시작
#------------------------------------------------------------

class FullyConnected_Layer(nn.Module):
    def __init__(self,
                 in_features,                # 입력 특성(차원) 수
                 out_features,               # 출력 특성(차원) 수
                 bias            = True,     # 활성화 전에 바이어스(bias)를 추가할지 여부
                 activation      = 'linear', # 활성화 함수: 'relu', 'lrelu', 'linear' 등
                 lr_multiplier   = 1,        # 학습률 스케일 (learning rate multiplier)
                 bias_init       = 0,        # 바이어스 초기값
                 ):
        
        super(FullyConnected_Layer, self).__init__()

        self.weight = nn.Parameter(
            torch.randn([out_features, in_features]) / lr_multiplier
        )

        self.bias = nn.Parameter(
            torch.full([out_features], np.float32(bias_init))
        ) if bias else None

        self.activation = activation
        self.weight_gain = lr_multiplier / np.sqrt(in_features) # Xiavier initialization
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight * self.weight_gain
        
        b = self.bias
        if b is not None and self.bias_gain != 1:
            b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = x.matmul(w.t())
            out = x + b.reshape([-1 if i == x.ndim-1 else 1 for i in range(x.ndim)])
        else:
            x = x.matmul(w.t())
            out = bias_act(x, b, act=self.activation, dim=x.ndim-1) 

        return out
    #---------------------------------
class Conv2dLayer(nn.Module):
    def __init__(self,
                 in_channels,                    # Number of input channels.
                 out_channels,                   # Number of output channels.
                 kernel_size,                    # Convolution kernel size.
                 bias            = True,         # Use bias?
                 activation      = 'linear',     # Activation function name.
                 up              = 1,            # Integer upsampling factor.
                 down            = 1,            # Integer downsampling factor.
                 groups          = 1,            # Number of groups for Grouped convolution.
                 ):
        super(Conv2dLayer, self).__init__()
        assert up == 1 or down == 1, "Can't do both up and down at the same time."
        assert kernel_size in [1, 3], "We only support kernel_size 1 or 3."

        self.activation = activation
        self.up = up
        self.down = down
        self.kernel_size = kernel_size
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.groups = groups
        
                
        if up != 1:  # Upsampling
            # ConvTranspose2d에서 weight.shape = [in_channels, out_channels, kH, kW]
            # (기존 F.conv_transpose2d도 같은 형식)
            assert out_channels % groups == 0 , "in_channels must be divided by groups"

            init_weight = torch.randn([in_channels, out_channels // groups, kernel_size, kernel_size])
            init_bias = torch.zeros([out_channels]) if bias else None

            if kernel_size == 1:
                padding = 0
                output_padding = 1  # 2배 upsampling
            else:  # kernel_size == 3
                padding = 1
                output_padding = 1 


            # nn.ConvTranspose2d 모듈 생성
            self.conv = nn.ConvTranspose2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride       = up,
                padding      = padding,
                output_padding = output_padding, 
                bias         = (init_bias is not None),
                groups = groups
            )

            # 우리가 초기화한 weight로 변경
            self.conv.weight = nn.Parameter(init_weight * self.weight_gain)
            if init_bias is not None:
                self.conv.bias = nn.Parameter(init_bias)

        else:
            # downsampling or normal conv case
            # Conv2d에서 weight.shape = [out_channels, in_channels, kH, kW]
            assert in_channels % groups == 0 , "in_channels must be divided by groups"

            init_weight = torch.randn([out_channels, in_channels // groups, kernel_size, kernel_size])
            init_bias = torch.zeros([out_channels]) if bias else None

            # kernel_size=1 → padding=0
            # kernel_size=3 → padding=1
            padding = 0 if (kernel_size == 1) else 1

            # 여기서는 그냥 일반 Conv2d
            self.conv = nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride       = down,
                padding      = padding,
                bias         = (init_bias is not None),
                groups = groups
            )

            self.conv.weight = nn.Parameter(init_weight * self.weight_gain)
            if init_bias is not None:
                self.conv.bias = nn.Parameter(init_bias)

    def forward(self, x):

        x = self.conv(x)
        out = bias_act(x, act=self.activation, dim=x.ndim-1) 

        return out
#---------------------------------
class FFN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, act_layer = nn.GELU, drop = 0.):
        super(FFN, self).__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        
        self.fc1 = FullyConnected_Layer(in_features = in_channels, out_features = hidden_channels, activation = 'lrelu')
        self.fc2 = FullyConnected_Layer(in_features = hidden_channels, out_features = out_channels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

#---------------------------------
class ModulatedConv2d(nn.Module): #Style을 배치/Out 채널별로 적용해주는 Conv class
    def __init__(self,
                 in_channels,                   # Number of input channels.
                 out_channels,                  # Number of output channels.
                 kernel_size,                   # Width and height of the convolution kernel.
                 style_dim,                     # dimension of the style code
                 demodulate=True,               # perfrom demodulation
                 up=1,                          # Integer upsampling factor.
                 down=1,                        # Integer downsampling factor.
                 ):
        
        
        super(ModulatedConv2d, self).__init__()

        self.demodulate = demodulate

        self.weight = torch.nn.Parameter(torch.randn([1, out_channels, in_channels, kernel_size, kernel_size]))
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.up = up
        self.down = down


        # nn.Conv 종류말고. F.conv를 사용해서 convolution연산 진행
        if up != 1:
            if kernel_size == 1:
                self.padding = 0
                self.output_padding = 1  # 2배 upsampling
            else:  # kernel_size == 3
                self.padding = 1
                self.output_padding = 1 
        elif down != 1:
            # Conv2d 에서 stride=down
            self.padding = 0 if (kernel_size == 1) else 1
            self.output_padding = None
        else:
            # up=1, down=1 이면 나중에 forward에서 kernel_size//2로 padding할 수도 있음
            self.padding = None
            self.output_padding = None
        
        self.affine = FullyConnected_Layer(style_dim, in_channels, bias = True, bias_init = 1)

    def forward(self, x, style):
        if x.ndim == 4:
            B, in_channels, H, W = x.shape
        else: # (B, L, C)
            B, L, in_channels = x.shape
            H, W = int(np.sqrt(L)), int(np.sqrt(L))
            
            x = token2feature(x, [H, H])

        style = self.affine(style).view(B, 1, in_channels, 1, 1)
        weight = self.weight * self.weight_gain * style # (B, Out, In, K, K)

        if self.demodulate: # 정규화
            decoefs = (weight.pow(2).sum(dim = [2, 3, 4]) + 1e-8).rsqrt() #(B, out_channels)
            weight = weight * decoefs.view(B, self.out_channels, 1, 1, 1) #(B, Out, In, K, K)

        
        x = x.reshape(-1, B * in_channels, H, W)

        if self.up != 1:
            # ConvTranspose2d
            # stride = up, padding = self.padding, output_padding = self.output_padding

            # ==== Transposed Conv ====
            # PyTorch는 weight shape = [in_channels, out_channels/groups, kH, kW] 로 기대.
            # 우린 현재 weight.shape = [B, Out, In, kH, kW].
            # 따라서 (B, Out, In, kH, kW) -> (B, In, Out, kH, kW)로 permute 후,
            # (B * In, Out, kH, kW): in channels을 batch로 쪼개서 grouped conv
            weight = weight.permute(0, 2, 1, 3, 4).contiguous()  # [B, In, Out, K, K]
            weight = weight.view(B * in_channels, self.out_channels, self.kernel_size, self.kernel_size)

            x = F.conv_transpose2d(
                x, weight,
                bias=None,
                stride=self.up,
                padding=self.padding,
                output_padding=self.output_padding,
                groups=B  # 배치마다 다른 weight를 적용하기 위해 groups=B
            )
        elif self.down != 1:
            # Conv2d with stride=down

            weight = weight.view(B * self.out_channels, in_channels, self.kernel_size, self.kernel_size)

            x = F.conv2d(
                x, weight,
                bias=None,
                stride=self.down,
                padding=self.padding,
                groups=B
            )
        else:
            # 일반 conv (stride=1)
            # kernel_size가 3이면 padding=1로 두는 게 일반적임. (원하는 출력 shape에 맞춰 조정 가능)

            weight = weight.view(B * self.out_channels, in_channels, self.kernel_size, self.kernel_size)

            p = self.kernel_size // 2
            x = F.conv2d(x, weight, bias=None, stride=1, padding=p, groups=B)

        # 최종 shape -> (B, C, H, W)
        _, _, H_out, W_out = x.shape
        out = x.view(B, self.out_channels, H_out, W_out) 


        return out

class StyleConv(torch.nn.Module): # ModulatedConv2d(스타일 반영 conv)에다가 noise를 추가해준 class
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        style_dim,                      # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        demodulate      = True,         # perform demodulation
    ):
        super().__init__()

        self.conv = ModulatedConv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    style_dim=style_dim,
                                    demodulate=demodulate,
                                    up=up,
                                    )

        self.use_noise = use_noise
        self.resolution = resolution
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([])) # noise 얼마나 강하게 반영할 것인지는 학습시키자

        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.activation = activation


    def forward(self, x, style, noise_mode='random', gain=1):
        x = self.conv(x, style)

        assert noise_mode in ['random', 'const', 'none']

        if self.use_noise: # noise 적용
            if noise_mode == 'random':
                H, W = x.size()[-2:]
                noise = torch.randn([x.shape[0], 1, H, W], device=x.device) * self.noise_strength

            if noise_mode == 'const':
                noise = self.noise_const * self.noise_strength

            if noise_mode == 'none':
                noise = torch.zeros([self.resolution, self.resolution])
            x = x + noise


        out = bias_act(x, self.bias, act=self.activation, dim = 1) # 마지막이 채널로 안 끝나서 dim 설정해줘야 함

        return out


class ToRGB(torch.nn.Module): # 1x1 kernel을 사용하는 ModulatedConv2d (+ Residual connection) 적용해서 RGB feature로 바꾸는 class
    def __init__(self,
                 in_channels,
                 out_channels,
                 style_dim,
                 kernel_size=1,
                 demodulate=False):
        super().__init__()

        self.conv = ModulatedConv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    style_dim=style_dim,
                                    demodulate=demodulate,
                                    )
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))


    def forward(self, x, style, skip=None):
        x = self.conv(x, style) # (B, C, H, W)
        out = bias_act(x, b = self.bias, dim = 1) # (B, C, H, W)

        if skip is not None:
            if skip.shape[-2:] != out.shape[-2:]: # skip이랑 크기 다르면 크기 맞춰주기
                skip = F.interpolate(skip, size=out.shape[-2:], mode='bilinear', align_corners=False)

            out = out + skip

        return out

    
class MappingNet(torch.nn.Module): # Generator랑 Discriminator에 쓰이는 class
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim

        # 각 단계마다의 feature dimension
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim] # [z_dim + embed_feature, layer_featueres, layer_featueres, ..., w_dim]

        if c_dim > 0:
            self.embed = FullyConnected_Layer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnected_Layer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'): # 각 단계 프로파일링(CPU & GPU 사용량 등을 확인)을 위한 함수 (https://jh-bk.tistory.com/20)
            if self.z_dim > 0:
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y # concat input & conditioning label

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta)) # torch.lerp(end, weight): interpolation

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        #Truncation Trick이라고 하는데, W를 W_{avg}에 가까워지도록 보장한다고 함 <- 안정적, but 다양성 감소
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)

        return x
    

class DisFromRGB(nn.Module): # Discriminator에 RGB feature를 줄 때 사용
    def __init__(self, in_channels, out_channels, activation):  # res = 2, ..., resolution_log2
        super().__init__()
        self.conv = Conv2dLayer(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                activation=activation,
                                )

    def forward(self, x):
        return self.conv(x)
    

class DisBlock(nn.Module): # Discriminator 블록
    def __init__(self, in_channels, out_channels, activation):  # res = 2, ..., resolution_log2
        super().__init__()
        self.conv0 = Conv2dLayer(in_channels=in_channels,
                                 out_channels=in_channels,
                                 kernel_size=3,
                                 activation=activation,
                                 )
        self.conv1 = Conv2dLayer(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 down=2,
                                 activation=activation,
                                 )
        self.skip = Conv2dLayer(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                down=2,
                                bias=False,
                             )

    def forward(self, x):
        skip = self.skip(x)
        x = self.conv0(x)
        x = self.conv1(x)
        out = skip + x

        return out
    

class MinibatchStdLayer(torch.nn.Module): # Mode Collapse 방지, 같은 배치 안에 속한 표본들간의 '표준편차'가 낮은 경우를 discriminator가 감지할 수 있도록 한다고 함(GPT)
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape

        # 그룹 수(mini batch) 정하기
        G = torch.min(torch.as_tensor(self.group_size),
                        torch.as_tensor(N)) if self.group_size is not None else N
        
        F = self.num_channels
        c = C // F  # 채널도 num_channels기준으로 쪼개기

        y = x.reshape(G, -1, F, c, H,
                      W)  # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)  # [GnFcHW] Subtract mean over group. [GnFcHW] - [nFcHW]
        y = y.square().mean(dim=0)  # [nFcHW]  Calc variance over group. sum((x - mean)^2) / # of x
        y = (y + 1e-8).sqrt()  # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])  # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)  # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)  # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)  # [NCHW]   Append to input as new channels.
        return x


class Conv2dLayerPartial(nn.Module):
    # 마스크가 주어지면 kernel 안에서 mask의 값이 1이면 conv를 통과한 하나의 픽셀에 대해서 전체 커널 개수 중에 몇 개가 mask인지 반영해서 weight 반영해주는 거라고 생각하면 됨 (mask에 아무것도 없으면 0으로 처리))
    def __init__(self,
                 in_channels,                    # Number of input channels.
                 out_channels,                   # Number of output channels.
                 kernel_size,                    # Width and height of the convolution kernel.
                 bias            = True,         # Apply additive bias before the activation function?
                 activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
                 up              = 1,            # Integer upsampling factor.
                 down            = 1,            # Integer downsampling factor.
                 trainable       = True,         # Update the weights of this layer during training?
                 ):
        super().__init__()
        self.conv = Conv2dLayer(in_channels, out_channels, kernel_size, bias, activation, up, down)

        self.weight_maskUpdater = torch.ones(1, 1, kernel_size, kernel_size)
        self.slide_winsize = kernel_size ** 2
        self.stride = down
        self.padding = kernel_size // 2 if kernel_size % 2 == 1 else 0

    def forward(self, x, mask=None):

        if mask is not None:
            with torch.no_grad():
                if self.weight_maskUpdater.type() != x.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(x)
                update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding) # mask 중에서 Kernel size 안에서 unmaksed point의 개수를 counting
                mask_ratio = self.slide_winsize / (update_mask + 1e-8) # unmask 부분이 적을수록 ration는 커진다 -> 많은 부분이 mask 되어있다는 것을 반영
                update_mask = torch.clamp(update_mask, 0, 1)  # update_mask는 0 ~ K^2로 정수단위임 -> 마스크된 부분이 있는지 여부만 궁금 -> 0 혹은 1로 clamping
                mask_ratio = torch.mul(mask_ratio, update_mask) # conv를 통과한 후의 feature map 상의 update_mask를 conv 통과전에 unmask 부분이 얼마나 많이 있는지 비율을 고려하여 업데이트

            x = self.conv(x)
            x = torch.mul(x, mask_ratio) # mask-update
            return x, update_mask
        else:
            x = self.conv(x) 
            return x, None
        
#---------------------------------
# DCN Tokenization: main structure
# We introdce DCN Tokenization, then we can replace Swin with standard Transformer
# We can reduce the TB withing a stage from 4 to 2.
#---------------------------------
class Tokenization(nn.Module):
    def __init__(self,
                 in_channels,                   # Number of channels of input 
                 out_channels,                  # Number of channels for ouput
                 kernel_size = 3,               # Namely.
                 padding = 1,                   # Padding of projection layer
                 stride = 1,                    # Stride of projection layer
                 deformable = False,            # Whether using deformabel convolution v3
                 method = 'default'             # method for normalization(i.e. LayerNormalization)                 
                 ):
        
        super(Tokenization, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deformable = deformable
        self.method = method

        assert method in ['bn', 'ln', None], f"We do not support {method} method."

        if deformable:
            self.proj = Deformable_Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding)
        else:
            self.proj = Conv2dLayer(in_channels, out_channels, kernel_size = kernel_size)
        
        if method == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif method == 'ln':
            self.norm = nn.LayerNorm(out_channels)


    def forward(self, x):

        # input shape: (B, C, H, W)
        x = self.proj(x)

        x = rearrange(x, 'B C H W -> B (H W) C') # (B, L, C)

        if self.method in ['bn', 'ln']:
            out = self.norm(x)
        else:
            out = F.normalize(x, p = 2.0, dim = -1)



        return out
    

class MHA(nn.Module):
    def __init__(self, 
                 in_channels,                       # Number of input channel
                 out_channels,                      # Number of output channel
                 head_num,                          # Number of head when doing MHSA operations
                 qk_scale = None,                   # scaling factor for computing attention score
                 qkv_bias = False,                  # Bias of CNN filters projecting feature map to Query, Key, and Value
                 attn_drop = 0,                     # Probability of dropout right before multiplying attention socres with Value matrix
                 proj_drop = 0,                     # Probability of dropout of projection, which projecting input to Query, Key, and Value matrix
                 kernel_size = 3,                   # Namely.
                 stride_kv = 1,                     # Stride of projection filter when projecting input to Key and Value
                 stride_q = 1,                      # Stride of projection filter when projecting input to Query
                 padding_kv = 1,                    # Padding of CNN projection when projecting input to Key and Valu
                 padding_q = 1,                     # Padding of CNN projection when projecting input to Query
                 method = None,                     # method of normalization (i.e. BatchNorm or LayerNorm)
                 deformable = False,                # Whether using DCN in Tokenization
                 with_cls_token = False,            # Whether use cls token
                 ):
        
        super(MHA, self).__init__()

        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.padding_q = padding_q
        self.padding_kv = padding_kv
        self.out_channels = out_channels
        self.head_num = head_num
        self.with_cls_token = with_cls_token
        self.head_dim = out_channels // head_num
        self.method = method
        self.deformable = deformable


        # Transformer 원 논문에 따라서 Scaling
        #self.scale = torch.sqrt(torch.tensor(out_channels / head_num))
        self.scale = qk_scale or self.head_dim ** -0.5

        # 일반 CNN Tokenization 거치면 mask update 해줘야 하기 위한 파라미터
        if not self.deformable:
            self.weight_maskUpdater = torch.ones(1, 1, kernel_size, kernel_size)
            self.slide_winsize = kernel_size ** 2


       




        # Overlapping-Tokenization with Deformable Convolution v3
        self.conv_proj_q = Tokenization(in_channels, out_channels, kernel_size = kernel_size,\
                                                  padding = padding_q, stride = stride_q,  deformable = deformable, method = method)
        
        self.conv_proj_k = Tokenization(in_channels, out_channels, kernel_size = kernel_size,\
                                                  padding = padding_kv, stride = stride_kv,  deformable = deformable, method = method)
        
        self.conv_proj_v = Tokenization(in_channels, out_channels, kernel_size = kernel_size,\
                                                  padding = padding_kv, stride = stride_kv,  deformable = deformable, method = method)
        
    

        # Query, Key, Value를 Hidden dimension으로 projection.
        self.proj_q = nn.Linear(in_channels, out_channels, bias = qkv_bias)
        self.proj_k = nn.Linear(in_channels, out_channels, bias = qkv_bias)
        self.proj_v = nn.Linear(in_channels, out_channels, bias = qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(in_channels, out_channels)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x, mask = None):
        # input shape: (B, C, H, W) ????????//
        # mask shape: (B, 1, H, W)

        B, C, H, W = x.shape

        q = self.conv_proj_q(x) # (B, L, C)
        k = self.conv_proj_k(x)
        v = self.conv_proj_v(x) 

        q = rearrange(self.proj_q(q), 'B L (H D) -> B H L D', H = self.head_num)
        k = rearrange(self.proj_k(k), 'B L (H D) -> B H L D', H = self.head_num)
        v = rearrange(self.proj_v(v), 'B L (H D) -> B H L D', H = self.head_num)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, L, L)


        if mask is not None:
            attn, mask = self._calculate_mask(attn, mask) # (B, 1, L) # MCA: attn_mask = (-inf / 0)
            attn_mask = mask.unsqueeze(-1).expand(-1, -1, -1, attn.size(-1))
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float(-100.0)).masked_fill(attn_mask == 1, float(0.0)) # (B, 1, L, L) / mask 토큰에 해당하는 row에 대해서는 attention = 0
            attn = attn + attn_mask # (B, H, L, L) 

            mask = mask.reshape(B, 1, H, W) # (B, 1, H, W)
                

        attn = self.softmax(attn)

        attention = attn @ v # output shae: (B, H, L, D)
        attention = rearrange(attention, 'B H L D -> B L (H D)')

        attention = self.proj(attention)
        attention = self.proj_drop(attention)

        return attention, mask
    
    def _calculate_mask(self, attn, mask, mask_size = None): #DCN or CNN 통과한 x랑 마스크 맞춰주기 위해서 convolution 연산해서 mask update
        if self.deformable:
            update_mask = self.conv_proj_v._forward_mask(mask)

        else:
            if mask is not None:
                with torch.no_grad():
                    if self.weight_maskUpdater.type() != attn.type():
                        self.weight_maskUpdater = self.weight_maskUpdater.to(attn)
                    update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride_kv, padding=self.padding_kv) # mask 중에서 Kernel size 안에서 unmaksed point의 개수를 counting
                    mask_ratio = self.slide_winsize / (update_mask + 1e-8) # unmask 부분이 적을수록 ration는 커진다 -> 많은 부분이 mask 되어있다는 것을 반영
                    update_mask = torch.clamp(update_mask, 0, 1)  # update_mask는 0 ~ K^2로 정수단위임 -> 마스크된 부분이 있는지 여부만 궁금 -> 0 혹은 1로 clamping
                    mask_ratio = torch.mul(mask_ratio, update_mask) # conv를 통과한 후의 feature map 상의 update_mask를 conv 통과전에 unmask 부분이 얼마나 많이 있는지 비율을 고려하여 업데이트

            
            attn = torch.mul(attn, mask_ratio) # mask-update
 


        update_mask = update_mask.masked_fill(update_mask == 0, 0).masked_fill(update_mask != 0, 1)

        update_mask = rearrange(update_mask, 'B 1 H W -> B 1 (H W)')

        return attn, update_mask
    

class TransformerBlock(nn.Module):
    def __init__(self,
                 in_channels,                       # Number of input channel
                 out_channels,                      # Number of output channel
                 head_num,                          # Number of head when doing MHSA operations
                 input_resolution,                  # Input resulotion
                 ffn_ratio,                         # scaling factoy that enlarging dimension in FFN block
                 frequency,                         # setting the frequency. ""
                 qk_scale = None,                   # scaling factor for computing attention score
                 qkv_bias = False,                  # Bias of CNN filters projecting feature map to Query, Key, and Value
                 attn_drop = 0.,                    # Probability of dropout right before multiplying attention socres with Value matrix
                 proj_drop = 0.,                    # Probability of dropout of projection, which projecting input to Query, Key, and Value matrix
                 stride_kv = 1,                     # Stride of projection filter when projecting input to Key and Value
                 stride_q = 1,                      # Stride of projection filter when projecting input to Query
                 padding_kv = 1,                    # Padding of CNN projection when projecting input to Key and Valu
                 padding_q = 1,                     # Padding of CNN projection when projecting input to Query
                 method = None,                     # method of normalization (i.e. BatchNorm or LayerNorm)
                 with_cls_token = False,            # Whether use cls token
                 act_layer=nn.GELU,                 # Activation layer. Default: nn.GELU
                 norm_layer=nn.LayerNorm,           # Normalization layer.  Default: nn.LayerNorm
                 use_ln = False,                    # Whether use LayerNormalization
                 ):
        
        super(TransformerBlock, self).__init__()


        self.in_channels = in_channels
        self.out_chaneels = out_channels
        self.head_num = head_num
        self.ffn_ratio = ffn_ratio
        self.input_resolution = input_resolution
        self.frequency = frequency
        self.use_ln = use_ln


        if frequency == "high":
            self.attn = MHA(in_channels, out_channels, head_num, deformable = False)
            if self.use_ln:
                self.mha_ln = nn.LayerNorm(in_channels, eps = 1e-6)
                self.ffn_ln  = nn.LayerNorm(out_channels, eps = 1e-6)
        elif frequency == 'low':
            self.attn = MHA(in_channels, out_channels, head_num, deformable = True)
        else:
            raise "Frequnecy is ambiguous. Please choose the frequency.[\'high\' or \'low\']"
        
        ffn_hidden_dim = int(out_channels * ffn_ratio)
        self.ffn = FFN(out_channels, ffn_hidden_dim, out_channels)

        if frequency == 'low':
            self.fuse = FullyConnected_Layer(out_channels * 2, out_channels) 
    

    def forward(self, x, x_size, low = None, mask = None):

        H, W = x_size
        B, L, C = x.shape
        assert L  == H*W, "input features wrong size"


        residual = x
        

        if mask is not None:
            mask = mask.view(B, 1, H, W)
    
        if self.frequency == 'low':
            x = rearrange(x, 'B (H W) C -> B C H W', H = H, W = W)
            x, mask = self.attn(x, mask) # (B, L, C)

            # FFN
            x = self.fuse(torch.cat([residual, x], dim = -1)) #concat with shortcut -> no residual connection -> learning low-frequency
            x = self.ffn(x)
            
        else: #use residual connection when learning high-frequency!
            if self.use_ln:
                x = self.mha_ln(x)
            x = rearrange(x, 'B (H W) C -> B C H W', H = H, W = W)

            x, mask = self.attn(x, mask)
            x = x + residual

            residual = x

            if self.use_ln:
                x = self.ffn_ln(x)
            x = self.ffn(x)

            x = x + residual

        return x, mask
    
#--------------------------------- 
# Basic Layer
#---------------------------------
class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, down=2):
        super().__init__()
        self.conv = Conv2dLayerPartial(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=3,
                                       activation='lrelu',
                                       down=down,
                                       )
        self.down = down

    def forward(self, x, x_size, mask=None):
        x = token2feature(x, x_size) # (B, C, H, W)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(x, mask)
        if self.down != 1:
            ratio = 1 / self.down
            x_size = (int(x_size[0] * ratio), int(x_size[1] * ratio))
        x = feature2token(x) # (B, L, C)
        if mask is not None:
            mask = feature2token(mask) 
        return x, x_size, mask
    
    
class PatchUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, up=2):
        super().__init__()
        self.conv = Conv2dLayerPartial(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=3,
                                       activation='lrelu',
                                       up=up,
                                       )
        self.up = up

    def forward(self, x, x_size, mask=None):
        x = token2feature(x, x_size)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(x, mask)
        if self.up != 1:
            x_size = (int(x_size[0] * self.up), int(x_size[1] * self.up))
        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask

#---------------------------------
# A basic Transformer with DCN Tokenization layer for one stage.
class BasicLayer(nn.Module):
    def __init__(self,
                 dim,                               # Number of input channels.
                 input_resolution,                  # Input resolution.
                 depth,                             # Number of blocks.
                 head_num,                          # Number of attention heads.
                 frequency,                         # setting the frequency.
                 ffn_ratio = 2.,                    # Ratio of mlp hidden dim to embedding dim.
                 qkv_bias = True,                   # If True, add a learnable bias to query, key, value. Default: True
                 qk_scale = None,                   # Override default qk scale of head_dim ** -0.5 if set.
                 proj_drop = 0.,                    # Dropout rate. Default: 0.0
                 attn_drop = 0.,                    # Attention dropout rate. Default: 0.0
                 stride_kv = 1,                     # Stride of projection filter when projecting input to Key and Value
                 stride_q = 1,                      # Stride of projection filter when projecting input to Query
                 padding_kv = 1,                    # Padding of CNN projection when projecting input to Key and Valu
                 padding_q = 1,                     # Padding of CNN projection when projecting input to Query
                 drop_path = 0.,                    # Stochastic depth rate. Default: 0.0 <- we won't use it
                 norm_layer = nn.LayerNorm,         # Normalization layer. Default: nn.LayerNorm
                 downsample = None,                 # Downsample layer at the end of the layer. Default: None
                 use_ln = False,                    # Whether use LayerNormalization
                 ):
        
        super(BasicLayer, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # patch merging layer between each stage
        if downsample is not None:
            # self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            self.downsample = downsample
        else:
            self.downsample = None

        # build blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(in_channels = dim, out_channels = dim, head_num = head_num,
                             input_resolution = input_resolution, ffn_ratio = ffn_ratio,
                             frequency = frequency, qk_scale = qk_scale, qkv_bias = qkv_bias,
                             attn_drop = attn_drop, proj_drop = proj_drop, stride_kv = stride_kv,
                             stride_q = stride_q, padding_kv = padding_kv, padding_q = padding_q,
                             norm_layer = norm_layer, use_ln = use_ln)
            for i in range(depth)])

        

        if frequency == 'low':
            self.dcn_fuse = Deformable_Conv2d(dim, dim)
            self.conv = Conv2dLayerPartial(in_channels=dim * 2, out_channels=dim, kernel_size=3, activation='lrelu')
        elif frequency == 'high':
            self.conv = Conv2dLayerPartial(in_channels=dim, out_channels=dim, kernel_size=3, activation='lrelu') # high-frequency도 고려


    def forward(self, x, x_size, mask=None, high = None):
        if self.downsample is not None:
            x, x_size, mask = self.downsample(x, x_size, mask)

        identity = x 

        for blk in self.blocks: # 여러 블록 통과
             x, mask = blk(x, x_size, mask)

        if mask is not None:
            mask = token2feature(mask, x_size)


        # Early Fusion
        if high is not None:  # high-frequency가 추가로 들어오면
            high = token2feature(high, x_size)
            high = self.dcn_fuse(high)
            high = feature2token(high)
            x = torch.cat([x, high], dim = -1) # (B, L, C)

        x, mask = self.conv(token2feature(x, x_size), mask)

        x = feature2token(x) + identity

        if mask is not None:
            mask = feature2token(mask)

        return x, x_size, mask
    

#---------------------------------
# Layer for Encoder & Decoder
#---------------------------------

class EncFromRGB(nn.Module):
    def __init__(self, in_channels, out_channels, activation):  # res = 2, ..., resolution_log2
        super().__init__()
        self.conv0 = Conv2dLayer(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                activation=activation,
                                )
        self.conv1 = Conv2dLayer(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                activation=activation,
                                )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        return x
    

class ConvBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, activation):  # res = 2, ..., resolution_log
        super().__init__()

        self.conv0 = Conv2dLayer(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 activation=activation,
                                 down=2,
                                 )
        self.conv1 = Conv2dLayer(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 activation=activation,
                                 )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        return x
    
#---------------------------------
class Encoder(nn.Module): # Conv layer를 연속으로 통과해서 encode
    def __init__(self, res_log2, img_channels, activation, patch_size=5, channels=16, drop_path_rate=0.1):
        super().__init__()

        self.resolution = []

        for idx, i in enumerate(range(res_log2, 3, -1)):  # from input size to 16x16
            res = 2 ** i
            self.resolution.append(res)
            if i == res_log2:
                block = EncFromRGB(img_channels * 2 + 1, nf(i), activation)
            else:
                block = ConvBlockDown(nf(i+1), nf(i), activation)
            setattr(self, 'EncConv_Block_%dx%d' % (res, res), block)

    def forward(self, x):
        out = {}
        for res in self.resolution:
            res_log2 = int(np.log2(res))
            x = getattr(self, 'EncConv_Block_%dx%d' % (res, res))(x)
            out[res_log2] = x

        return out

         
class ToStyle(nn.Module): # Getting Style vector 아마도
    def __init__(self, in_channels, out_channels, activation, drop_rate):
        super().__init__()
        self.conv = nn.Sequential(
                Conv2dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, activation=activation, down=2),
                Conv2dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, activation=activation, down=2),
                Conv2dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, activation=activation, down=2),
                )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = FullyConnected_Layer(in_features=in_channels,
                                      out_features=out_channels,
                                      activation=activation)

    def forward(self, x):
        # Conv -> Conv -> Conv -> Average Pooling -> FC
        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x.flatten(start_dim=1))

        return x
    

#---------------------------------
# Decoder 부분에 처음 block-> conv로 맵핑하고  Encoder Feature 추가 (U-Net), modulated conv 통과, rgb로 mapping
class DecBlockFirstV2(nn.Module):
    
    def __init__(self, res, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):
        super().__init__()
        self.res = res

        self.conv0 = Conv2dLayer(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=3,
                                activation=activation,
                                )
        
        # ModulatedConv2d(스타일 반영 conv)에다가 noise를 추가해준 class
        self.conv1 = StyleConv(in_channels=in_channels,
                              out_channels=out_channels,
                              style_dim=style_dim,
                              resolution=2**res,
                              kernel_size=3,
                              use_noise=use_noise,
                              activation=activation,
                              demodulate=demodulate,
                              )
        self.toRGB = ToRGB(in_channels=out_channels,
                           out_channels=img_channels,
                           style_dim=style_dim,
                           kernel_size=1,
                           demodulate=False,
                           )

    def forward(self, x, ws, gs, E_features, noise_mode='random'):
        # x = self.fc(x).view(x.shape[0], -1, 4, 4)
        x = self.conv0(x)
        x = x + E_features[self.res] # Skip-connection (U-Net) <- Encoder의 해당 resolution에 해당하는 feature 가져옴
        style = get_style_code(ws[:, 0], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, 1], gs)
        img = self.toRGB(x, style, skip=None)

        return x, img

            

class DecBlock(nn.Module):
    def __init__(self, res, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):  # res = 4, ..., resolution_log2
        super(DecBlock, self).__init__()
        self.res = res

        # Upscaling with StyleConv
        self.conv0 = StyleConv(in_channels=in_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               up=2,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.conv1 = StyleConv(in_channels=out_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.toRGB = ToRGB(in_channels=out_channels,
                           out_channels=img_channels,
                           style_dim=style_dim,
                           kernel_size=1,
                           demodulate=False,
                           )

    def forward(self, x, img, ws, gs, E_features, noise_mode='random'):
        style = get_style_code(ws[:, self.res * 2 - 9], gs)
        x = self.conv0(x, style, noise_mode=noise_mode)
        x = x + E_features[self.res]
        style = get_style_code(ws[:, self.res * 2 - 8], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, self.res * 2 - 7], gs)
        img = self.toRGB(x, style, skip=img)

        return x, img


class Decoder(nn.Module):
    def __init__(self, res_log2, activation, style_dim, use_noise, demodulate, img_channels):
        super(Decoder, self).__init__()

        # 연속적인 Decoder block 쌓기
        self.Dec_16x16 = DecBlockFirstV2(4, nf(4), nf(4), activation, style_dim, use_noise, demodulate, img_channels)
        for res in range(5, res_log2 + 1):
            setattr(self, 'Dec_%dx%d' % (2 ** res, 2 ** res),
                    DecBlock(res, nf(res - 1), nf(res), activation, style_dim, use_noise, demodulate, img_channels))
        self.res_log2 = res_log2

    def forward(self, x, ws, gs, E_features, noise_mode='random'):
        x, img = self.Dec_16x16(x, ws, gs, E_features, noise_mode=noise_mode)
        for res in range(5, self.res_log2 + 1):
            block = getattr(self, 'Dec_%dx%d' % (2 ** res, 2 ** res))
            x, img = block(x, img, ws, gs, E_features, noise_mode=noise_mode)

        # 최종적으로 image scale로 바꾼 이미지를 return
        return img

class DecStyleBlock(nn.Module): # 바로 아래 FirstStage 클래스에서 사용
    def __init__(self, res, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):
        super(DecStyleBlock, self).__init__()
        self.res = res

        self.conv0 = StyleConv(in_channels=in_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               up=2,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.conv1 = StyleConv(in_channels=out_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.toRGB = ToRGB(in_channels=out_channels,
                           out_channels=img_channels,
                           style_dim=style_dim,
                           kernel_size=1,
                           demodulate=False,
                           )

    def forward(self, x, img, style, skip, noise_mode='random'):
        x = self.conv0(x, style, noise_mode=noise_mode)
        x = x + skip
        x = self.conv1(x, style, noise_mode=noise_mode)
        img = self.toRGB(x, style, skip=img)

        return x, img

class DecHighBlock(nn.Module): # 바로 아래 FirstStage 클래스에서 사용
    def __init__(self, in_channels, out_channels, activation):
        super(DecHighBlock, self).__init__()

        self.conv0 = Conv2dLayer(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               up=2,
                               activation=activation,
                               )
        
        self.conv1 = Conv2dLayer(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               up=1,
                               activation=activation,
                               )

    def forward(self, x, skip):
        x = self.conv0(x)
        x = x + skip
        x = self.conv1(x)

        return x

#---------------------------------
# Main Stages
#--------------------------------- 
class FirstStage(nn.Module):
    def __init__(self,
                 img_channels,
                 img_resolution=256,
                 dim=180,
                 w_dim=512,
                 use_noise=False,
                 demodulate=True, 
                 activation='lrelu',
                 use_edge = False,          # whether input masked edge images
                 use_line = False,          # whether input masked line images
                 use_ln = False,            # Whether use LayerNormalization
                 few_high = False,          # Whether use fewer blocks in high-freqeuncy
                 ):
        super().__init__()
        res = 64

        if use_edge and use_line: # 기존 그림과 다르게 그냥 처음부터 masked_image, mask, masked_edge, masked_line 정보 다 모아서 각 frequency에 넣어준다
            self.conv_first = Conv2dLayerPartial(in_channels=img_channels+3, out_channels=dim, kernel_size=3, activation=activation) # gray scale의 edge & line도 추가
        elif use_edge or use_line:
            self.conv_first = Conv2dLayerPartial(in_channels=img_channels+2, out_channels=dim, kernel_size=3, activation=activation)
        else:
            self.conv_first = Conv2dLayerPartial(in_channels=img_channels+1, out_channels=dim, kernel_size=3, activation=activation)

        self.enc_conv = nn.ModuleList()
        self.use_edge = use_edge
        self.use_line = use_line
        self.few_high = few_high

        down_time = int(np.log2(img_resolution // res)) # 몇 번 resolution 줄일 것인지

        for i in range(down_time):  # from input size to 64
            self.enc_conv.append(
                    Conv2dLayerPartial(in_channels=dim, out_channels=dim, kernel_size=3, down=2, activation=activation)
                )
        # from 64 -> 16 -> 64
        #depths = [2, 3, 4, 3, 2]
        depths = [1, 1, 1, 1, 1]
        ratios = [1, 1/2, 1/2, 2, 2]
        head_num = 6

        if self.few_high:
            high_depths = [1, 2, 2, 2, 1]
        else:
            high_depths = [i for i in depths]



        self.low = nn.ModuleList()
        self.high = nn.ModuleList()

        for i, depth in enumerate(zip(depths, high_depths)):
            res = int(res * ratios[i])

            if ratios[i] < 1: # feature map이 작아지면
                merge_high = PatchMerging(dim, dim, down = int(1/ratios[i]))
                merge_low = PatchMerging(dim, dim, down = int(1/ratios[i]))

            elif ratios[i] > 1: # feature map이 커지면
                merge_high = PatchUpsampling(dim, dim, up = ratios[i])
                merge_low = PatchUpsampling(dim, dim, up = ratios[i])
            else:
                merge = None

            self.high.append(
                BasicLayer(dim=dim, input_resolution=[res, res], depth=depth[1],
                           head_num = head_num, downsample=merge, frequency = 'high', use_ln = use_ln)
            )

            self.low.append(
                BasicLayer(dim=dim, input_resolution=[res, res], depth=depth[0],
                           head_num = head_num, downsample=merge, frequency = 'low', use_ln = False)
            )


        # global style
        down_conv = []
        for i in range(int(np.log2(16))):
            down_conv.append(Conv2dLayer(in_channels=dim, out_channels=dim, kernel_size=3, down=2, activation=activation))

        down_conv.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.down_conv = nn.Sequential(*down_conv)
        self.to_style = FullyConnected_Layer(in_features=dim, out_features=dim*2, activation=activation)
        self.ws_style = FullyConnected_Layer(in_features=w_dim, out_features=dim, activation=activation)
        self.to_square = FullyConnected_Layer(in_features=dim, out_features=16*16, activation=activation)
        self.get_edge_line = FullyConnected_Layer(in_features = dim, out_features = 2, activation = activation)

        style_dim = dim * 3
        self.dec_conv = nn.ModuleList()
        high_res = res # high_frequency를 위한 res도 있어야 함
        for i in range(down_time):  # from 64 to input size
            res = res * 2
            self.dec_conv.append(DecStyleBlock(res, dim, dim, activation, style_dim, use_noise, demodulate, img_channels))

        # high-frequency도 input size로 만들어줘야 BCE ㅣoss 줄 수 있음
        self.high_dec_conv = nn.ModuleList()
        for i in range(down_time):  # from 64 to input size
            res = res * 2
            self.high_dec_conv.append(DecHighBlock(dim, dim, activation))


    def forward(self, images_in, masks_in, ws, noise_mode='random', edge = None, line = None):

        #(masks_in - 0.5): 마스크를 -0.5~0.5 범위로 이동 -> 평균 0으로 만들어서 네트워크가 마스크 정보를 더 잘 학습하도록 도와줌
        #(images_in * masks_in): 이미지에서 알려진(=마스크=1) 부분만 남긴 텐서 <- 얘는 image_in이 이미 normalize되어 있어서 이동 안하는 듯
        if (edge is not None and line is not None) and (self.use_edge and self.use_line): # edge와 line에 대한 정보도 살아있어야 함
            x = torch.cat([masks_in - 0.5, images_in * masks_in, edge * masks_in, line * masks_in], dim = 1)
        elif edge is not None and self.use_edge:
            x = torch.cat([masks_in - 0.5, images_in * masks_in, edge * masks_in], dim = 1)
        elif line is not None and self.use_line:
            x = torch.cat([masks_in - 0.5, images_in * masks_in, line * masks_in], dim = 1)
        else: 
            x = torch.cat([masks_in - 0.5, images_in * masks_in], dim=1)

        #-------------------------------
        # Encoder 통과 
        #-------------------------------
        skips = [] # 나중에 skip-connection (U-Net)을 위해서 저장하기 위해 list 선언 (for Decoder)
        high_skips = [] # skip for high-frequency

        x, mask = self.conv_first(x, masks_in)  
        skips.append(x)
        for i, block in enumerate(self.enc_conv):  # input size to 64
            x, mask = block(x, mask)
            if i != len(self.enc_conv) - 1:
                skips.append(x)


        #-------------------------------
        # 메인 스테이지 통과 
        #-------------------------------
        x_size = x.size()[-2:]
        x = feature2token(x) # (B, L, C)

        mask = feature2token(mask) # (B, L, C)
        mid = len(self.low) // 2
        for i, block in enumerate(zip(self.high, self.low)):  # 64 to 16 to 64
            if i < mid: #donwsampling
                if i == 0:
                    high, _, high_mask = block[0](x, x_size, mask = mask)
                    low, x_size, low_mask = block[1](x, x_size, mask = mask, high = high)
                    
                else:
                    high, _, high_mask = block[0](high, x_size, mask = high_mask)
                    low, x_size, low_mask = block[1](low, x_size, mask = low_mask, high = high)

                skips.append(low)
                high_skips.append(high)

                
            elif i > mid:
                # upsampling 할 떄는, 블록단위 residual connection 해주는 듯 <- U-Net처럼
                high, _, high_mask = block[0](high, x_size, mask = None)
                low, x_size, low_mask = block[1](low, x_size, mask = None, high = high)

                low = low + skips[mid - i]
                high = high + high_skips[mid - i]
            else:
                #가운데 Stage여서 style 관련된 연산 추가 <- 이거 논문 overall architecture figure 보면 이해 됨ㄴ
                high, _, high_mask = block[0](high, x_size, mask = None)
                low, x_size, low_mask = block[1](low, x_size, mask = None, high = high)

                # mul_map: Dropout으로 랜덤 부분만 0이 될 수 있는 텐서(=0.5)
                mul_map = torch.ones_like(low) * 0.5
                mul_map = F.dropout(mul_map, training=True)

                ws = self.ws_style(ws[:, -1]) # 가장 최근(마지막) style vector를 사용하기 위해서 dim으로 projection
                add_n = self.to_square(ws).unsqueeze(1) # to 16*16
                add_n = F.interpolate(add_n, size=high.size(1), mode='linear', align_corners=False).squeeze(1).unsqueeze(-1) # (B, L, 1)

                # 0.5 비율로 x는 유지 나머지는 style vector를 반영
                low = low * mul_map + add_n * (1 - mul_map)
                gs = self.to_style(self.down_conv(token2feature(low, x_size)).flatten(start_dim=1)) # to dim*2

                # 가장 응축된 단계(mid)로부터 global style vector를 찾음 (ws랑 global style(gs))
                style = torch.cat([gs, ws], dim=1)

        x = token2feature(low, x_size).contiguous() # (B, C, H, W)
        img = None

        #-------------------------------
        # Decoder 통과
        #-------------------------------
        for i, block in enumerate(self.dec_conv):
            low, img = block(low, img, style, skips[len(self.dec_conv)-i-1], noise_mode=noise_mode)

        high_size = int(np.sqrt(high.shape[1]))
        high = token2feature(high, x_size = [high_size, high_size]) # (B, C, H, W)

        for i, block in enumerate(self.high_dec_conv):
            high = block(high, skips[len(self.high_dec_conv) - i - 1])

        # ensemble
        # generated image의 mask area랑 unmasked area of feature of GT image랑 합치기 
        img = img * (1 - masks_in) + images_in * masks_in

        high_size = high.shape[-2:]
        high = self.get_edge_line(feature2token(high)) #(B, 2, H, W)
        high = token2feature(high, high_size)

        

        if (edge is not None) or (line is not None): # GT 있으면 unmask부분은 GT로 설정, 없으면 그냥 high 그대로 반환
            if line is None:
                line = torch.zeros_like(edge)
            if edge is None:
                edge = torch.zeros_like(line)

            high  = high * (1 - masks_in) + torch.cat([edge, line], dim = 1) * masks_in  # (B, 2, H, W)

        return img, high



#-------------------------------        
class SynthesisNet(nn.Module):
    def __init__(self,
                 w_dim,                         # Intermediate latent (W) dimensionality.
                 img_resolution,                # Output image resolution.
                 img_channels   = 3,            # Number of color channels.
                 channel_base   = 32768,        # Overall multiplier for the number of channels.
                 channel_decay  = 1.0,
                 channel_max    = 512,          # Maximum number of channels in any layer.
                 activation     = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
                 drop_rate      = 0.5,
                 use_noise      = True,
                 demodulate     = True,
                 use_edge       = False,        # whether input masked edge images
                 use_line       = False,        # whether input masked line images
                 use_ln         = False,        # Whether use LayerNormalization
                 few_high       = False,        # Whether use fewer blocks in high-freqeuncy
                 ):
        super().__init__()
        resolution_log2 = int(np.log2(img_resolution))
        assert img_resolution == 2 ** resolution_log2 and img_resolution >= 4

        self.num_layers = resolution_log2 * 2 - 3 * 2
        self.img_resolution = img_resolution
        self.resolution_log2 = resolution_log2
        self.use_edge = use_edge
        self.use_line = use_line

        # first stage
        self.first_stage = FirstStage(img_channels, img_resolution=img_resolution, w_dim=w_dim, use_noise=False,
                                      demodulate=demodulate, use_edge = use_edge, use_line = use_line,
                                      use_ln = use_ln, few_high = few_high)

        # second stage
        self.enc = Encoder(resolution_log2, img_channels, activation, patch_size=5, channels=16)
        self.to_square = FullyConnected_Layer(in_features=w_dim, out_features=16*16, activation=activation)
        self.to_style = ToStyle(in_channels=nf(4), out_channels=nf(2) * 2, activation=activation, drop_rate=drop_rate)
        style_dim = w_dim + nf(2) * 2
        self.dec = Decoder(resolution_log2, activation, style_dim, use_noise, demodulate, img_channels)

    def forward(self, images_in, masks_in, ws, noise_mode='random', return_stg1=False, edge = None, line = None):
        out_stg1, high_freq = self.first_stage(images_in, masks_in, ws, noise_mode=noise_mode, edge = edge, line = line)

        # encoder
        x = images_in * masks_in + out_stg1 * (1 - masks_in) # 마스크 부분은 만들어낸 걸로 
        x = torch.cat([masks_in - 0.5, x, images_in * masks_in], dim=1)
        E_features = self.enc(x)

        fea_16 = E_features[4]# 16x16 features
        mul_map = torch.ones_like(fea_16) * 0.5
        mul_map = F.dropout(mul_map, training=True)
        add_n = self.to_square(ws[:, 0]).view(-1, 16, 16).unsqueeze(1) # 처음 style을 square
        add_n = F.interpolate(add_n, size=fea_16.size()[-2:], mode='bilinear', align_corners=False)
        fea_16 = fea_16 * mul_map + add_n * (1 - mul_map)
        E_features[4] = fea_16 # mid stage에 넣어줄 encoder의 feature를 업데이트 (ws 반영해줘서)

        # global style
        gs = self.to_style(fea_16) # mid stage의 style vector

        # decoder
        img = self.dec(fea_16, ws, gs, E_features, noise_mode=noise_mode)

        # ensemble
        img = img * (1 - masks_in) + images_in * masks_in
        

        if not return_stg1:
            return img, high_freq
        else:
            return img, out_stg1, high_freq


#-------------------------------
# Generator
#-------------------------------
class Generator(nn.Module):
    def __init__(self,
                 z_dim,                     # Input latent (Z) dimensionality, 0 = no latent.
                 c_dim,                     # Conditioning label (C) dimensionality, 0 = no label.
                 w_dim,                     # Intermediate latent (W) dimensionality.
                 img_resolution,            # resolution of generated image
                 img_channels,              # Number of input color channels.
                 use_edge = False,          # whether input masked edge images
                 use_line = False,          # whether input masked line images
                 use_ln = False,            # Whether use LayerNormalization
                 few_high = False,          # Whether use fewer blocks in high-freqeuncy
                 synthesis_kwargs = {},     # Arguments for SynthesisNetwork.
                 mapping_kwargs   = {},     # Arguments for MappingNetwork.
                 ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.synthesis = SynthesisNet(w_dim=w_dim,
                                      img_resolution=img_resolution,
                                      img_channels=img_channels,
                                      use_edge = use_edge,
                                      use_line = use_line,
                                      use_ln = use_ln,
                                      few_high = few_high,
                                      **synthesis_kwargs)
        
        # Getting the 'ws'
        self.mapping = MappingNet(z_dim=z_dim,
                                  c_dim=c_dim,
                                  w_dim=w_dim,
                                  num_ws=self.synthesis.num_layers,
                                  **mapping_kwargs)

    def forward(self, images_in, masks_in, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False,
                noise_mode='random', return_stg1=False, edge = None, line = None):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                          skip_w_avg_update=skip_w_avg_update)

        if not return_stg1:
            img, high_freq = self.synthesis(images_in, masks_in, ws, noise_mode=noise_mode, edge = edge, line = line)
            return img, high_freq
        else:
            img, out_stg1, high_freq = self.synthesis(images_in, masks_in, ws, noise_mode=noise_mode, return_stg1=True, edge = edge, line = line)
            return img, out_stg1, high_freq


#-------------------------------
# Discriminator
#-------------------------------
class Discriminator(torch.nn.Module):
    def __init__(self,
                 c_dim,                        # Conditioning label (C) dimensionality.
                 img_resolution,               # Input resolution.
                 img_channels,                 # Number of input color channels.
                 channel_base       = 32768,    # Overall multiplier for the number of channels.
                 channel_max        = 512,      # Maximum number of channels in any layer.
                 channel_decay      = 1,
                 cmap_dim           = None,     # Dimensionality of mapped conditioning label, None = default.
                 activation         = 'lrelu',
                 mbstd_group_size   = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
                 mbstd_num_channels = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
                 ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        resolution_log2 = int(np.log2(img_resolution))
        assert img_resolution == 2 ** resolution_log2 and img_resolution >= 4
        self.resolution_log2 = resolution_log2

        if cmap_dim == None:
            cmap_dim = nf(2)
        if c_dim == 0:
            cmap_dim = 0
        self.cmap_dim = cmap_dim

        if c_dim > 0:
            self.mapping = MappingNet(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None)

        Dis = [DisFromRGB(img_channels+1, nf(resolution_log2), activation)]
        for res in range(resolution_log2, 2, -1):
            Dis.append(DisBlock(nf(res), nf(res-1), activation))

        if mbstd_num_channels > 0:
            Dis.append(MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels))
        Dis.append(Conv2dLayer(nf(2) + mbstd_num_channels, nf(2), kernel_size=3, activation=activation))
        self.Dis = nn.Sequential(*Dis)

        self.fc0 = FullyConnected_Layer(nf(2)*4**2, nf(2), activation=activation)
        self.fc1 = FullyConnected_Layer(nf(2), 1 if cmap_dim == 0 else cmap_dim)

        # for 64x64
        # 생성된 이미지를 구별하기 위한 레이어들
        Dis_stg1 = [DisFromRGB(img_channels+1, nf(resolution_log2) // 2, activation)]
        for res in range(resolution_log2, 2, -1):
            Dis_stg1.append(DisBlock(nf(res) // 2, nf(res - 1) // 2, activation))

        if mbstd_num_channels > 0:
            Dis_stg1.append(MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels))
        Dis_stg1.append(Conv2dLayer(nf(2) // 2 + mbstd_num_channels, nf(2) // 2, kernel_size=3, activation=activation))
        self.Dis_stg1 = nn.Sequential(*Dis_stg1)

        self.fc0_stg1 = FullyConnected_Layer(nf(2) // 2 * 4 ** 2, nf(2) // 2, activation=activation)
        self.fc1_stg1 = FullyConnected_Layer(nf(2) // 2, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, images_in, masks_in, images_stg1, c):
        # 원래 이미지 feature
        x = self.Dis(torch.cat([masks_in - 0.5, images_in], dim=1))
        x = self.fc1(self.fc0(x.flatten(start_dim=1)))

        # 생성된 이미지 feature
        x_stg1 = self.Dis_stg1(torch.cat([masks_in - 0.5, images_stg1], dim=1))
        x_stg1 = self.fc1_stg1(self.fc0_stg1(x_stg1.flatten(start_dim=1)))

        # condition vector c를 곱해주는 듯
        if self.c_dim > 0:
            cmap = self.mapping(None, c)

        if self.cmap_dim > 0:
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
            x_stg1 = (x_stg1 * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        # 최종적으로 GT Image와 Generated Image의 score(예측값)을 각각 도출
        return x, x_stg1


if __name__ == '__main__':
    # mac 환경이라서 그냥 cpu
    device = torch.device('cpu')
    batch = 2
    res = 512
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=512, img_channels=3, use_edge = True,
                  use_line = False, use_ln = True, few_high = False).to(device)
    
    D = Discriminator(c_dim=0, img_resolution=res, img_channels=3).to(device)
    img = torch.randn(batch, 3, res, res).to(device)
    mask = torch.randn(batch, 1, res, res).to(device)
    edge = torch.randn(batch, 1, res, res).to(device)
    line = torch.randn(batch, 1, res, res).to(device)
    z = torch.randn(batch, 512).to(device)
    G.eval()

    # def count(block):
    #     return sum(p.numel() for p in block.parameters()) / 10 ** 6
    # print('Generator', count(G))
    # print('discriminator', count(D))

    def count(block):
        return sum(p.numel() for p in block.parameters()) / 10 ** 6
    print(f'Params: {count(G)} milions')


    with torch.no_grad():
        img, img_stg1, high_freq = G(img, mask, z, None, return_stg1=True, edge = edge, line = None)
        #(B, 2, H, W)
        # index 0: edge
        # index 1: line

    print('output of G:', img.shape, img_stg1.shape, high_freq.shape)
    score, score_stg1 = D(img, mask, img_stg1, None)
    print('output of D:', score.shape, score_stg1.shape)


