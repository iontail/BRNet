import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#------------------
# 기존적인 함수 초기화
#------------------

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



def bias_act(x, b = None, act = 'linear', normalize = None, dim = None, gain = 1):
    # Normalize layer도 고려

    if dim is not None and b is not None:
        b_shape = [1 for _ in range(len(x.shape))]
        b_shape[dim] = x.size(dim)

        b = b.reshape(*b_shape)

    if b is not None:
        x = x + b


    activation, _ = get_activation(act)

    if normalize is not None:
        channels = x.size(1)

        assert normalize in [None, 'bn', 'ln'], f"unsupported normalization name. please set the argument one of [None, 'bn, 'ln']."

        if normalize == 'bn':
            out = nn.BatchNorm2d(channels)(x)
        elif normalize == 'ln':
            out = nn.LayerNorm(channels)(x)
    

    out = activation()(out)

    #out = out * (gain * act_gain)
    #우리는 gain 고려 안 할 거임 (하면 좋긴하다)

    return out


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    # x * sqrt(x^2.mean)
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


#------------------
# basic layers
#------------------

class FullyConnected_Layer(nn.Module):
    def __init__(self,
                 in_channels,                # 입력 특성(차원) 수
                 out_channels,               # 출력 특성(차원) 수
                 bias            = True,     # 활성화 전에 바이어스(bias)를 추가할지 여부
                 activation      = 'relu', # 활성화 함수: 'relu', 'lrelu', 'linear' 등
                 lr_multiplier   = 1,        # 학습률 스케일 (learning rate multiplier)
                 bias_init       = 0,        # 바이어스 초기값
                 ):
        
        super(FullyConnected_Layer, self).__init__()

        self.weight = nn.Parameter(
            torch.randn([out_channels, in_channels]) / lr_multiplier
        )

        self.bias = nn.Parameter(
            torch.full([out_channels], np.float32(bias_init))
        ) if bias else None

        self.activation = activation
        self.weight_gain = lr_multiplier / np.sqrt(in_channels) # Xiavier initialization
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
    

class Conv2dLayer(nn.Module): # custom conv2d layer
    def __init__(self,
                 in_channels,                    # Number of input channels.
                 out_channels,                   # Number of output channels.
                 kernel_size,                    # Convolution kernel size.
                 bias            = False,         # Use bias?
                 activation      = 'linear',     # Activation function name.
                 normalize       = None,         # Use normalization?
                 up              = 1,            # Integer upsampling factor.
                 down            = 1,            # Integer downsampling factor.
                 groups          = 1,            # Number of groups for Grouped convolution.
                 ):
        super(Conv2dLayer, self).__init__()
        assert up == 1 or down == 1, "Can't do both up and down at the same time."
        assert kernel_size in [1, 3], "We only support kernel_size 1 or 3."

        self.activation = activation
        self.normalize = normalize
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
        out = bias_act(x, act=self.activation, normalize = self.normalize, dim=x.ndim-1) 

        return out