import os
import torch
import torch.nn as nn

from data.config import cfg

from bioreflectnet.models.layers.basic_layers import *
from models.modules.l2norm import L2Norm
from bioreflectnet.models.layers.DSFD_basic_modules import DistillKL, fem_module, add_extras
from bioreflectnet.models.functions.detection import Detect
from models.modules.priorbox import PriorBox
from bioreflectnet.models.BRNet import BRNet


class DSFD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
        boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, base, extras, fem, head1, head2, num_classes, cfg):
        super(DSFD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.vgg = base
        self.config = cfg
        self.ort_func = nn.CosineSimilarity(dim=1, eps=1e-8)


        self.L2Normof1 = L2Norm(256, 10)
        self.L2Normof2 = L2Norm(512, 8)
        self.L2Normof3 = L2Norm(512, 5)

        self.extras = nn.ModuleList(extras)
        self.fpn_topdown = nn.ModuleList(fem[0])
        self.fpn_latlayer = nn.ModuleList(fem[1])

        self.fpn_fem = nn.ModuleList(fem[2])

        self.L2Normef1 = L2Norm(256, 10)
        self.L2Normef2 = L2Norm(512, 8)
        self.L2Normef3 = L2Norm(512, 5)

        self.loc_pal1 = nn.ModuleList(head1[0])
        self.conf_pal1 = nn.ModuleList(head1[1])

        self.loc_pal2 = nn.ModuleList(head2[0])
        self.conf_pal2 = nn.ModuleList(head2[1])

        ##############################
        # self.ref를 두 개로 나눠서 1024로 projection 시키고 semi-orthogonal loss 주기 위해서 loss 계산
        # 그리고 3 channel로 projection 해야 할 듯
        # 1. 문제는 이렇게 하면 loss를 마지막 레이어(conv7)의 output에 주는데, DSFD는 각 레이어의 feature에서 detection을 실행하게 되는데
        # 한 곳에만 Loss를 주게되면 문제점이 있지 않을까?
        # 2. 그리고 1st PAL과 2nd PAL 중에서 어는 PAL의 마지막 아웃풋에 loss를 적용해주어야 할까?
        # small object가 중요하면 1st PAL에 적용
        # "t low-level features are more suitable for small faces, we assign smaller anchor sizes in the first shot, 
        # and use larger sizes in the second shot." from DSFD - https://arxiv.org/pdf/1512.02325.pdf'
        ##############################
        # the reflectance decoding branch
        
        

        self.KL = DistillKL(T=4.0)

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(self.config)

    def _upsample_prod(self, x, y): # y크기로 upsample한 다음에 y과 element-product
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') * y


    # during training, the model takes the paired images, and their pseudo GT illumination maps from the Retinex Decom Net
    def forward(self, x, x_light, I, I_light):
        size = x.size()[2:]
        pal1_sources = list()
        pal2_sources = list()
        loc_pal1 = list()
        conf_pal1 = list()
        loc_pal2 = list()
        conf_pal2 = list()

        # 밝은 이미지 통과
        features_light, darklevel_light, R_light = self.vgg(x_light, reflectance = False) # 모든 레이어 통과

        # 어두운 이미지 통과
        features, darklevel, R_dark = self.vgg(x, reflectance = False)
    

        # Interchange for "Redecomposition"
        x_dark_2 = (I * R_light).detach()
        x_light_2 = (I_light * R_dark).detach()

        # Redecomposition
        R_dark_2 = self.vgg(x_dark_2, reflectance = True)[2]
        R_light_2 = self.vgg(x_light_2, reflectance = True)[2]

        # --------------------------------
        # mutual feature alignment loss
        # --------------------------------
        x_light = x_light.flatten(start_dim=2).mean(dim=-1)
        x_dark = x_dark.flatten(start_dim=2).mean(dim=-1)
        x_light_2 = x_light_2.flatten(start_dim=2).mean(dim=-1)
        x_dark_2 = x_dark_2.flatten(start_dim=2).mean(dim=-1)

        loss_mutual = self.config.WEIGHT.MC * (self.KL(x_light, x_dark) + self.KL(x_dark, x_light) + \
                             self.KL(x_light_2, x_dark_2) + self.KL(x_dark_2, x_light_2))


        # the following is the rest of the original detection pipeline

        for i, feature in enumerate(features): #!!! 1stage 값 포함할것인지 의논해야함!!!!!!! 
            if i < 3:
                feature = getattr(self, f'L2Normef{i+1}')(feature)
                pal1_sources.append(features)

            elif i == 3:
                x = feature # 다음 레이어에 통과시키기 위해 복사
                pal1_sources.append(feature)

        # considering extra layers and cache source layer outputs
                

        for k in range(2):
            x = F.relu(self.extras[k](x), inplace=True)
        of5 = x
        pal1_sources.append(of5)

        for k in range(2, 4):
            x = F.relu(self.extras[k](x), inplace=True)
        of6 = x
        pal1_sources.append(of6)

        conv7 = F.relu(self.fpn_topdown[0](of6), inplace=True)

        # --------------------------------
        # Semi Orthogonal Regularity Loss (미완성)
        # --------------------------------
        # SOTR loss from "https://github.com/cuiziteng/ICCV_MAET.git" of https://arxiv.org/abs/2205.03346
        loss_sotr = self.config.WEIGHT.SOTR * torch.mean(torch.abs(self.ort_func (grads['light_grad'].view(batch_size,-1), grads['dark_grad'].view(batch_size,-1))))+\
                            0.5*torch.mean(1 - torch.abs(self.ort_func(grads['light_grad'].view(batch_size,-1), grads['light_grad'].view(batch_size,-1)))) +\
                            0.5*torch.mean(1 - torch.abs(self.ort_func(grads['dark_grad'].view(batch_size,-1), grads['dark_grad'].view(batch_size,-1))))


        x = F.relu(self.fpn_topdown[1](conv7), inplace=True)
        conv6 = F.relu(self._upsample_prod(
            x, self.fpn_latlayer[0](of5)), inplace=True)

        x = F.relu(self.fpn_topdown[2](conv6), inplace=True)
        convfc7_2 = F.relu(self._upsample_prod(
            x, self.fpn_latlayer[1](pal1_sources[3])), inplace=True) #of4

        x = F.relu(self.fpn_topdown[3](convfc7_2), inplace=True)
        conv5 = F.relu(self._upsample_prod(
            x, self.fpn_latlayer[2](pal1_sources[2])), inplace=True)

        x = F.relu(self.fpn_topdown[4](conv5), inplace=True)
        conv4 = F.relu(self._upsample_prod(
            x, self.fpn_latlayer[3](pal1_sources[1])), inplace=True)

        x = F.relu(self.fpn_topdown[5](conv4), inplace=True)
        conv3 = F.relu(self._upsample_prod(
            x, self.fpn_latlayer[4](pal1_sources[0])), inplace=True)

        ef1 = self.fpn_fem[0](conv3)
        ef1 = self.L2Normef1(ef1)
        ef2 = self.fpn_fem[1](conv4)
        ef2 = self.L2Normef2(ef2)
        ef3 = self.fpn_fem[2](conv5)
        ef3 = self.L2Normef3(ef3)
        ef4 = self.fpn_fem[3](convfc7_2)
        ef5 = self.fpn_fem[4](conv6)
        ef6 = self.fpn_fem[5](conv7)

        pal2_sources = (ef1, ef2, ef3, ef4, ef5, ef6)
        for (x, l, c) in zip(pal1_sources, self.loc_pal1, self.conf_pal1): # -> (B, H, W, C)
            loc_pal1.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf_pal1.append(c(x).permute(0, 2, 3, 1).contiguous())

        for (x, l, c) in zip(pal2_sources, self.loc_pal2, self.conf_pal2):
            loc_pal2.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf_pal2.append(c(x).permute(0, 2, 3, 1).contiguous())

        features_maps = []
        for i in range(len(loc_pal1)):
            feat = []
            feat += [loc_pal1[i].size(1), loc_pal1[i].size(2)]
            features_maps += [feat]

        loc_pal1 = torch.cat([o.view(o.size(0), -1)
                              for o in loc_pal1], 1)
        conf_pal1 = torch.cat([o.view(o.size(0), -1)
                               for o in conf_pal1], 1)

        loc_pal2 = torch.cat([o.view(o.size(0), -1)
                              for o in loc_pal2], 1)
        conf_pal2 = torch.cat([o.view(o.size(0), -1)
                               for o in conf_pal2], 1)

        priorbox = PriorBox(size, features_maps, self.config.PRIORBOX, pal=1)
        self.priors_pal1 = priorbox.forward().detach()

        priorbox = PriorBox(size, features_maps, self.config.PRIORBOX, pal=2)
        self.priors_pal2 = priorbox.forward().detach()

        if self.phase == 'test':
            output = self.detect.forward(
                loc_pal2.view(loc_pal2.size(0), -1, 4),
                self.softmax(conf_pal2.view(conf_pal2.size(0), -1,
                                            self.num_classes)),  # conf preds
                self.priors_pal2.type(type(x.data))
            )

        else:
            output = (
                loc_pal1.view(loc_pal1.size(0), -1, 4),
                conf_pal1.view(conf_pal1.size(0), -1, self.num_classes),
                self.priors_pal1,
                loc_pal2.view(loc_pal2.size(0), -1, 4),
                conf_pal2.view(conf_pal2.size(0), -1, self.num_classes),
                self.priors_pal2)

        # packing the outputs from the reflectance decoder:
        return output, [R_dark, R_light, R_dark_2, R_light_2], [darklevel, darklevel_light], loss_mutual, loss_sotr

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            mdata = torch.load(base_file,
                               map_location=lambda storage, loc: storage)

            epoch = 50
            self.load_state_dict(mdata)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
        return epoch

    def xavier(self, param):
        nn.init.xavier_uniform(param)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)
            m.bias.data.zero_()

        if isinstance(m, nn.ConvTranspose2d):
            self.xavier(m.weight.data)
            m.bias.data.zero_()

        if isinstance(m, nn.ConvTranspose2d):
            self.xavier(m.weight.data)
            if 'bias' in m.state_dict().keys():
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data[...] = 1
            m.bias.data.zero_()



extras_cfg = [256, 'S', 512, 128, 'S', 256]

fem_cfg = [256, 512, 512, 1024, 512, 256]


def multibox(vgg, extra_layers, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [11, 14, 17, -2]

    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                  num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels,
                                 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels,
                                  num_classes, kernel_size=3, padding=1)]
    return (loc_layers, conf_layers)


def build_net_DSFD(phase, num_classes=2):
    base = Bio_ReflectNet(backbone = 'vgg')
    extras = add_extras(extras_cfg, 1024)
    head1 = multibox(base, extras, num_classes)
    head2 = multibox(base, extras, num_classes)
    fem = fem_module(fem_cfg)
    return DSFD(phase, base, extras, fem, head1, head2, num_classes, cfg = cfg)


