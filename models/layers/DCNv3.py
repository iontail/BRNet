import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

class DCNv3(nn.Module):
    """
    Reference code url: https://github.com/Chenfeng1271/Adaptive-deformable-convolution/blob/master/modeling/deformable_conv/deform_conv_v3.py
        Args:

        modulation          # If True, Modulated Deformable Convolution (DCN v2)

        this class support kernel size with 3 and 1 only.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=2,
                 bias=None, modulation=True, adaptive_d=True):
        super(DCNv3, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.adaptive_d = adaptive_d
        self.modulation = modulation

        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        # Kernel의 각 포인트(K^2)마다 x & y에 대한 값 (2*)
        self.p_conv = nn.Conv2d(in_channels, 2 * kernel_size**2, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        nn.init.constant_(self.p_conv.bias, 0)

        # offset에 대해서는 learning rate의 0.1배로 학습
        self.p_conv.weight.register_hook(lambda grad: grad * 0.1)
        self.p_conv.bias.register_hook(lambda grad: grad * 0.1)

        if self.modulation:
            # depth-wise weight를 주기 위한 modulation scalar
            self.m_conv = nn.Conv2d(in_channels, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride, bias=False)
            nn.init.constant_(self.m_conv.weight, 0.5)
            self.m_conv.weight.register_hook(lambda grad: grad * 0.1)

        if self.adaptive_d:
            self.ad_conv = nn.Conv2d(in_channels, kernel_size, kernel_size=3, padding=1, stride=stride, bias=False)
            nn.init.constant_(self.ad_conv.weight, 1)
            self.ad_conv.weight.register_hook(lambda grad: grad * 0.1)

    def forward(self, x):
        offset = self.p_conv(x)  # output shape: (B, 2 * K^2, H, W)

        # Adaptive dilation 처리
        if self.adaptive_d:
            ad_base = self.ad_conv(x)  # output shape: (B, K, H, W)
            ad_base = 1 - torch.sigmoid(ad_base)
            ad = ad_base.repeat(1, 2 * self.kernel_size, 1, 1) * self.dilation  # output shape: (B, 2 * K^2, H, W)

            ad_m = (ad_base - 0.5) * 2  # -1 ~ 1 범위로 변환
            ad_m = ad_m.repeat(1, self.kernel_size, 1, 1) * self.dilation  # output shape: (B, K^2, H, W)

            self.ad_m = ad_m.detach()

        # Modulation 처리
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))  # output shape: (B, K^2, H, W)

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2  # N = K^2

        # 필요시 zero-padding
        if self.padding:
            x = self.zero_padding(x)

        # Adaptive_d가 True이면 오프셋에 ad를 곱한 결과를 구한다
        if self.adaptive_d:
            p = self._get_p(offset, dtype, ad)
        else:
            p = self._get_p(offset, dtype, None)

        p = p.contiguous().permute(0, 2, 3, 1)  # shape: (B, H, W, 2N)

        # Bilinear interpolation 위한 좌표 계산
        q_lt = torch.floor(p)
        q_rb = q_lt + 1

        self.q_lt = torch.cat([
            torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),
            torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)
        ], dim=-1).long()

        self.q_rb = torch.cat([
            torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
            torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)
        ], dim=-1).long()

        self.q_lb = torch.cat([self.q_lt[..., :N], self.q_rb[..., N:]], -1)
        self.q_rt = torch.cat([self.q_rb[..., :N], self.q_lt[..., N:]], -1)

        # Bilinear interpolation weights
        g_lt = (1 + (self.q_lt[..., :N].type_as(p) - p[..., :N])) * \
               (1 + (self.q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (self.q_rb[..., :N].type_as(p) - p[..., :N])) * \
               (1 - (self.q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (self.q_lb[..., :N].type_as(p) - p[..., :N])) * \
               (1 - (self.q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (self.q_rt[..., :N].type_as(p) - p[..., :N])) * \
               (1 + (self.q_rt[..., N:].type_as(p) - p[..., N:]))

        # 4-pixel interpolation
        x_q_lt = self._get_x_q(x, self.q_lt, N)
        x_q_rb = self._get_x_q(x, self.q_rb, N)
        x_q_lb = self._get_x_q(x, self.q_lb, N)
        x_q_rt = self._get_x_q(x, self.q_rt, N)

        x_offset = (
            g_lt.unsqueeze(dim=1) * x_q_lt +
            g_rb.unsqueeze(dim=1) * x_q_rb +
            g_lb.unsqueeze(dim=1) * x_q_lb +
            g_rt.unsqueeze(dim=1) * x_q_rt
        )

        # Modulation 적용
        if self.modulation:
            if self.adaptive_d:
                m = m * ad_m

            m = m.contiguous().permute(0, 2, 3, 1).unsqueeze(dim=1)
            self.m = m.detach()
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        # 최종 Conv
        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _forward_mask(self, x):

        if self.padding:
            x = self.zero_padding(x)
        
        dtype = self.offset.data.type()
        ks = self.kernel_size
        N = self.offset.size(1) // 2


        # -----------------------
        # binear interpolation weight
        # -----------------------
        g_lt = (1 + (self.q_lt[..., :N].type_as(self.p) - self.p[..., :N])) * \
            (1 + (self.q_lt[..., N:].type_as(self.p) - self.p[..., N:]))

        g_rb = (1 - (self.q_rb[..., :N].type_as(self.p) - self.p[..., :N])) * \
           (1 - (self.q_rb[..., N:].type_as(self.p) - self.p[..., N:]))
        
        g_lb = (1 + (self.q_lb[..., :N].type_as(self.p) - self.p[..., :N])) * \
           (1 - (self.q_lb[..., N:].type_as(self.p) - self.p[..., N:]))
        
        g_rt = (1 - (self.q_rt[..., :N].type_as(self.p) - self.p[..., :N])) * \
           (1 + (self.q_rt[..., N:].type_as(self.p) - self.p[..., N:]))
        
        # -----------------------
        # 4 pixel interpolation
        # -----------------------
        #self._get_x_q: 실제로 (q_*) 좌표에서 x의 픽셀값을 샘플링
        x_q_lt = self._get_x_q(x, self.q_lt, N) # (B, C, H, W, N)
        x_q_rb = self._get_x_q(x, self.q_rb, N)
        x_q_lb = self._get_x_q(x, self.q_lb, N)
        x_q_rt = self._get_x_q(x, self.q_rt, N)
        

        # result of bilinear interpolation
        # 각 채널간에는 똑같은 offset 적용
        x_offset = (
            g_lt.unsqueeze(dim = 1) * x_q_lt +
            g_rb.unsqueeze(dim = 1) * x_q_rb +
            g_lb.unsqueeze(dim = 1) * x_q_lb +
            g_rt.unsqueeze(dim = 1) * x_q_rt 
        )


        if self.modulation:

            # x_offset 채널 개수만큼 broadcast
            m = torch.cat([self.m for _ in range(x_offset.size(1))], dim = 1)


            # 최종적으로 x_offset에 곱해서 위치별 weight 조정
            x_offset *= m


        # -----------------------
        # 최종 Conv
        # -----------------------
        # (B, C, H, W, K^2) -> (B, C x K^2, H, W)
        x_offset = self._reshape_x_offset(x_offset, ks) 
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):

        

        """
        커널크기를 기준으로 상대좌표를 meshgrid로 생성
        (1, 2N, 1, 1) 모양으로 반환
        p_n_x:tensor([[-1, -1, -1],
                    [ 0,  0,  0],
                    [ 1,  1,  1]])
            

        p_n_y:tensor([[-1,  0,  1],
                    [-1,  0,  1],
                    [-1,  0,  1]])
            
        """
        k = self.kernel_size
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(k-1)//2, (k-1)//2 + 1),
            torch.arange(-(k-1)//2, (k-1)//2 + 1),
            indexing = 'ij'
        )

        p_n_x, p_n_y = p_n_x.type(dtype), p_n_y.type(dtype)
        
        # x와 y를 이어붙여서 [2*K*K]로 만들고, (1, 2N, 1, 1)으로 reshape
        p_n = torch.cat([p_n_x.flatten(), p_n_y.flatten()], dim = 0)
        p_n = p_n.reshape(1, 2*N, 1, 1).type(dtype)

        return p_n


    def _get_p_0(self, N, h, w, dtype):
        """
        h, w와 stride를 기준으로 각 위치의 기본좌표 (p_0_x, p_0_y)를 (1, 2N, h, w)에 맞게 생성
        """
        
        
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride),
            indexing = 'ij'
        )

        
        """
        줄어들기 이전의 feature map에서의 위치를 설정하는 메소드이기 때문에 stride를 기준으로 진행
        즉, 각 feature가 줄어들기 이전 feature map에서 어디에서 온 feature인지 기준으로 잠는 base 좌표를 구한다
        """



        p_0_x = torch.flatten(p_0_x).reshape(1, 1, h, w).repeat(1, N, 1, 1) # shape: (1, N, h, w)
        p_0_y = torch.flatten(p_0_y).reshape(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], dim = 1).type(dtype) # shape: (1, 2N, h, w)

        return p_0
        

    def _get_p(self, offset, dtype, ad_offset):
        """
        최종 offset 좌표 p = p_0 + p_n + offset ( + ad_offset * p_n)
        offset: (B, 2N, H, W)
        """

        B, _, H, W = offset.shape
        N = offset.size(1) // 2

        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(N, H, W, dtype)

        # p_0 + p_n => base deform
        # + offset => learnable offset
        # + ad_offset * p_n => adaptiev dialtion
        if (self.adaptive_d) and (ad_offset is not None):
            p = p_0 + p_n + offset + ad_offset * p_n
        else:
            p = p_0 + p_n + offset

        return p

    def _get_x_q(self, x, q, N):
       """
       binear interpolation을 위해서 q에 해당하는 인덱스에서 x 값을 gather
       x: (B, C, H', W')
       q: (B, H, W, 2N) (각 위치별 x, y인덱스)
       """ 
       B, H, W, _ = q.size()
       padded_w = x.size(3)
       C = x.size(1)

    
       x = x.contiguous().view(B, C, -1) # shape: (B, C, H' * W')
       
       # q[..., :N] = x 방향 인덱스, q[..., N:] = y방향 인덱스
       # index = x_idx * width + y_idx
       index = q[..., :N] * padded_w + q[..., N:] # shape: (B, H, W, N)

       index = index.contiguous().unsqueeze(dim = 1).expand(-1, C, -1, -1, -1) # shape: (B, C,  H, W, N)
       index = index.contiguous().view(B, C, -1).to(torch.int64) # shape: (B, C, H * W * N)

       x_offset = x.gather(dim = -1, index = index).contiguous().view(B, C, H, W, N)

       return x_offset


    
    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        """
        x_offset: (B, C, H, W, N) (N = ks^2라고 가정)
        -> (B, C, H*ks, W*ks) 형태로 재배열
        """
        B, C, H, W, N = x_offset.size()
        x_offset = rearrange(x_offset, 'B C H W (KH KW) -> B C (H KH) (W KW)', KH = ks, KW = ks)
        
        return x_offset



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNv3(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1).to(device)
    x = torch.randn(1, 3, 32, 32).to(device)
    output = model(x)
    print("Output shape:", output.shape)