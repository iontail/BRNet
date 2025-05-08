import os
import torch
import torch.nn.functional as F
import torch.nn as nn

from data.config import cfg

from layers.basic_layers import *
from modules.l2norm import L2Norm
from layers.DSFD_basic_modules import DistillKL, fem_module, add_extras
from functions.detection import Detect
from modules.priorbox import PriorBox
from BRNet import BRNet
from layers.PhotoReceptor_module import GainBlock, ReflectiveFeedback # Import for type checking




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
        self.brnet = base
        self.cfg = cfg

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

        # define for our new semi-orthogonal regularity loss
        self.ort_func = nn.CosineSimilarity(dim=1, eps=1e-8)

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
            self.detect = Detect(self.cfg)

    def _upsample_prod(self, x, y): # y크기로 upsample한 다음에 y과 element-product
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) * y

    # method for inference
    def test_forward(self, x):
        size = x.size()[2:]
        pal1_sources = list()
        pal2_sources = list()
        loc_pal1 = list()
        conf_pal1 = list()
        loc_pal2 = list()
        conf_pal2 = list()

        # Use BRNet for feature extraction instead of direct VGG backbone
        features, darklevel, R, _ = self.brnet(x, only_disentangle=False)
        
        # Extract features from BRNet outputs
        # Starting from the first feature after initial processing (same as in forward method)
        for i, feature in enumerate(features[1:]):  # starting from output of 16th layer of BRNet
            if i < 3:
                s = getattr(self, f'L2Normef{i+1}')(feature)
                pal1_sources.append(s)
            elif i == 3:
                x = feature  # Save for passing to next layers
                pal1_sources.append(feature)

        # Apply extra layers as in original test_forward
        for k in range(2):
            x = F.relu(self.extras[k](x), inplace=True)
        of5 = x
        pal1_sources.append(of5)
        
        for k in range(2, 4):
            x = F.relu(self.extras[k](x), inplace=True)
        of6 = x
        pal1_sources.append(of6)

        # Feature Pyramid Network (FPN) processing
        conv7 = F.relu(self.fpn_topdown[0](of6), inplace=True)

        x = F.relu(self.fpn_topdown[1](conv7), inplace=True)
        conv6 = F.relu(self._upsample_prod(
            x, self.fpn_latlayer[0](of5)), inplace=True)

        x = F.relu(self.fpn_topdown[2](conv6), inplace=True)
        convfc7_2 = F.relu(self._upsample_prod(
            x, self.fpn_latlayer[1](pal1_sources[3])), inplace=True)

        x = F.relu(self.fpn_topdown[3](convfc7_2), inplace=True)
        conv5 = F.relu(self._upsample_prod(
            x, self.fpn_latlayer[2](pal1_sources[2])), inplace=True)

        x = F.relu(self.fpn_topdown[4](conv5), inplace=True)
        conv4 = F.relu(self._upsample_prod(
            x, self.fpn_latlayer[3](pal1_sources[1])), inplace=True)

        x = F.relu(self.fpn_topdown[5](conv4), inplace=True)
        conv3 = F.relu(self._upsample_prod(
            x, self.fpn_latlayer[4](pal1_sources[0])), inplace=True)

        # Feature Enhancement Module (FEM)
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
        
        # Apply localization and confidence layers
        for (x, l, c) in zip(pal1_sources, self.loc_pal1, self.conf_pal1):
            loc_pal1.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf_pal1.append(c(x).permute(0, 2, 3, 1).contiguous())

        for (x, l, c) in zip(pal2_sources, self.loc_pal2, self.conf_pal2):
            loc_pal2.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf_pal2.append(c(x).permute(0, 2, 3, 1).contiguous())

        # Collect feature map information for prior box generation
        features_maps = []
        for i in range(len(loc_pal1)):
            feat = []
            feat += [loc_pal1[i].size(1), loc_pal1[i].size(2)]
            features_maps += [feat]
        
        # Flatten and concatenate outputs
        loc_pal1 = torch.cat([o.view(o.size(0), -1)
                            for o in loc_pal1], 1)
        conf_pal1 = torch.cat([o.view(o.size(0), -1)
                            for o in conf_pal1], 1)

        loc_pal2 = torch.cat([o.view(o.size(0), -1)
                            for o in loc_pal2], 1)
        conf_pal2 = torch.cat([o.view(o.size(0), -1)
                            for o in conf_pal2], 1)

        # Generate prior boxes
        priorbox = PriorBox(size, features_maps, self.cfg, pal=1)
        self.priors_pal1 = priorbox.forward().detach()

        priorbox = PriorBox(size, features_maps, self.cfg, pal=2)
        self.priors_pal2 = priorbox.forward().detach()

        # Generate detection output
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
                
        return output, R


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
        features_light, darklevel_light, R_light, _ = self.brnet(x_light, only_disentangle = False) # 모든 레이어 통과
        x_light = features_light[0]

        # 어두운 이미지 통과
        features_dark, darklevel_dark, R_dark, _ = self.brnet(x, only_disentangle = True)
        x_dark = features_dark[0] 
    

        # Interchange for "Redecomposition"
        # change the Reflectance
        x_dark_2 = (I * R_light).detach()
        x_light_2 = (I_light * R_dark).detach()

        # Redecomposition
        R_dark_2 = self.brnet(x_dark_2, only_disentangle = True)[2] # getting the reflectance
        R_light_2 = self.brnet(x_light_2, only_disentangle = True)[2]

        # --------------------------------
        # mutual feature alignment loss
        # --------------------------------
        x_light = x_light.flatten(start_dim=2).mean(dim=-1) # (B, C, 1, 1)
        x_dark = x_dark.flatten(start_dim=2).mean(dim=-1)
        x_light_2 = x_light_2.flatten(start_dim=2).mean(dim=-1)
        x_dark_2 = x_dark_2.flatten(start_dim=2).mean(dim=-1)

        loss_mutual = self.cfg.WEIGHT.MC * (self.KL(x_light, x_dark) + self.KL(x_dark, x_light) + \
                             self.KL(x_light_2, x_dark_2) + self.KL(x_dark_2, x_light_2))


        # the following is the rest of the original detection pipeline
        for i, feature in enumerate(features_light[1:]): # starting from output of 16th layer of BRNet
            if i < 3:
                s = getattr(self, f'L2Normef{i+1}')(feature)
                pal1_sources.append(s)

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
        loss_sotr = 0
        """
        loss_sotr = self.config.WEIGHT.SOTR * torch.mean(torch.abs(self.ort_func (grads['light_grad'].view(batch_size,-1), grads['dark_grad'].view(batch_size,-1))))+\
                            0.5*torch.mean(1 - torch.abs(self.ort_func(grads['light_grad'].view(batch_size,-1), grads['light_grad'].view(batch_size,-1)))) +\
                            0.5*torch.mean(1 - torch.abs(self.ort_func(grads['dark_grad'].view(batch_size,-1), grads['dark_grad'].view(batch_size,-1))))
        """

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

        priorbox = PriorBox(size, features_maps, self.cfg, pal=1)
        self.priors_pal1 = priorbox.forward().detach()

        priorbox = PriorBox(size, features_maps, self.cfg, pal=2)
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
        return output, [R_dark, R_light, R_dark_2, R_light_2], [darklevel_dark, darklevel_light], loss_mutual, loss_sotr

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
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.ConvTranspose2d):
            self.xavier(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

        # Zero Convolution!
        # Specific zero initialization for Conv2d layers within GainBlock and ReflectiveFeedback
        # It overrides any previous initialization for Conv2d layers inside these blocks.
        if isinstance(m, (GainBlock, ReflectiveFeedback)):
            for layer_in_block in m.layers: # m.layers is the nn.Sequential
                if isinstance(layer_in_block, nn.Conv2d):
                    layer_in_block.weight.data.zero_()
                    if layer_in_block.bias is not None:
                        layer_in_block.bias.data.zero_()



extras_cfg = [256, 'S', 512, 128, 'S', 256]

fem_cfg = [256, 512, 512, 1024, 512, 256]


def multibox(extra_layers, num_classes):
    """
    Creates the localization and confidence prediction layers for DSFD with BRNet backbone.
    
    Args:
        extra_layers: Additional layers added after the backbone
        num_classes: Number of classes for classification
        
    Returns:
        Tuple of (loc_layers, conf_layers)
    """
    loc_layers = []
    conf_layers = []
    
    # Determine correct channel counts based on BRNet implementation
    # Looking at BRNet.py we can see the actual channel counts in the returned features
    brnet_feature_channels = [256, 512, 512, 1024]
    
    # Create location and confidence layers for BRNet features
    for channels in brnet_feature_channels:
        loc_layers += [nn.Conv2d(channels, 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(channels, num_classes, kernel_size=3, padding=1)]
    
    # Add layers for extra features
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, num_classes, kernel_size=3, padding=1)]
    
    return (loc_layers, conf_layers)

def build_net_DSFD(phase, num_classes=2):
    base = BRNet(3, cfg)
    extras = add_extras(extras_cfg, 1024)
    head1 = multibox(extras, num_classes)
    head2 = multibox(extras, num_classes)
    fem = fem_module(fem_cfg)
    return DSFD(phase, base, extras, fem, head1, head2, num_classes, cfg = cfg)

if __name__ == '__main__':
    import torch

    print("DSFD_BRNet.py 실행 테스트 시작...")

    from data.config import cfg

    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        return total_params, trainable_params, non_trainable_params


    # 모델 빌드 파라미터 정의
    input_height = cfg.INPUT_SIZE
    input_width = cfg.INPUT_SIZE
    print(f"config.py에서 로드된 입력 크기 사용: {input_height}x{input_width}")
    
    # Batch size 및 채널 수 정의
    batch_size = 2
    channels = 3
    
    # 더미 입력 텐서 생성
    dummy_input = torch.randn(batch_size, channels, input_height, input_width)
    dummy_input_light = torch.randn(batch_size, channels, input_height, input_width)
    dummy_I = torch.randn(batch_size, 1, input_height, input_width)  # Illumination map
    dummy_I_light = torch.randn(batch_size, 1, input_height, input_width)  # Light illumination map
    
    print(f"더미 입력 텐서 생성 완료.")
    print(f"- 어두운 이미지 형태: {dummy_input.shape}")
    print(f"- 밝은 이미지 형태: {dummy_input_light.shape}")
    print(f"- 어두운 이미지 조명 맵 형태: {dummy_I.shape}")
    print(f"- 밝은 이미지 조명 맵 형태: {dummy_I_light.shape}")

    # 1. TEST 모드에서 모델 테스트
    print("\n===== TEST 모드 테스트 =====")
    test_phase = 'test'
    num_classes = cfg.NUM_CLASSES
    
    print(f"DSFD 모델 빌드 중... (phase='{test_phase}', num_classes={num_classes})")
    try:
        test_model = build_net_DSFD(phase=test_phase, num_classes=num_classes)
        test_model.eval()
        print("모델 빌드 성공.")

        total_params, trainable_params, non_trainable_params = count_parameters(test_model)
        print(f"  - 총 파라미터 수: {total_params:,}")
        print(f"  - 학습 가능한 파라미터 수: {trainable_params:,}")
        print(f"  - 학습 불가능한 파라미터 수: {non_trainable_params:,}")
    except Exception as e:
        print(f"모델 빌드 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # test_forward 실행
    print("test_forward 메서드 실행 중...")
    try:
        with torch.no_grad():
            outputs, R_dark = test_model.test_forward(dummy_input)
            print("test_forward 실행 성공.")

            # 출력 정보 인쇄
            print("\n--- 출력 정보 ---")
            print(f"test_forward의 'outputs' 타입: {type(outputs)}")
            if isinstance(outputs, torch.Tensor):
                print(f"  Detections 텐서 형태: {outputs.shape}")
            elif isinstance(outputs, tuple):
                print(f"  Output이 튜플입니다, 길이: {len(outputs)}")
                for i, item in enumerate(outputs):
                    if isinstance(item, torch.Tensor):
                        print(f"    outputs[{i}] 형태: {item.shape}")
                    else:
                        print(f"    outputs[{i}]: {type(item)}")
            else:
                print(f"  Detections: {type(outputs)}")

            print(f"test_forward의 'R_dark' 타입: {type(R_dark)}")
            if isinstance(R_dark, torch.Tensor):
                print(f"  R_dark 텐서 형태: {R_dark.shape}")
            else:
                print(f"  R_dark: {type(R_dark)}")
    except Exception as e:
        print(f"test_forward 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

    # 2. TRAIN 모드에서 모델 테스트
    print("\n===== TRAIN 모드 테스트 =====")
    train_phase = 'train'
    
    print(f"DSFD 모델 빌드 중... (phase='{train_phase}', num_classes={num_classes})")
    try:
        train_model = build_net_DSFD(phase=train_phase, num_classes=num_classes)
        train_model.train()
        print("모델 빌드 성공.")
    except Exception as e:
        print(f"모델 빌드 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # forward 메서드 실행
    print("forward 메서드 실행 중...")
    try:
        # train 모드에서는 forward 메서드에 4개의 입력이 필요함
        outputs, reflectances, darklevels, loss_mutual, loss_sotr = train_model(
            dummy_input, dummy_input_light, dummy_I, dummy_I_light
        )
        print("forward 실행 성공.")

        # 출력 정보 인쇄
        print("\n--- 출력 정보 ---")
        print(f"forward의 'outputs' 타입: {type(outputs)}")
        if isinstance(outputs, tuple):
            print(f"  Output은 튜플입니다, 길이: {len(outputs)}")
            for i, item in enumerate(outputs):
                if isinstance(item, torch.Tensor):
                    print(f"    outputs[{i}] 형태: {item.shape}")
                else:
                    print(f"    outputs[{i}]: {type(item)}")
                    
        print(f"forward의 'reflectances' 타입: {type(reflectances)}")
        if isinstance(reflectances, list):
            print(f"  리플렉턴스 리스트 길이: {len(reflectances)}")
            for i, item in enumerate(reflectances):
                if isinstance(item, torch.Tensor):
                    print(f"    reflectances[{i}] 형태: {item.shape}")
                else:
                    print(f"    reflectances[{i}]: {type(item)}")
                    
        print(f"forward의 'darklevels' 타입: {type(darklevels)}")
        if isinstance(darklevels, list):
            print(f"  다크레벨 리스트 길이: {len(darklevels)}")
            for i, item in enumerate(darklevels):
                if isinstance(item, torch.Tensor):
                    print(f"    darklevels[{i}] 형태: {item.shape}")
                else:
                    print(f"    darklevels[{i}]: {type(item)}")
                    
        print(f"loss_mutual 값: {loss_mutual}")
        print(f"loss_sotr 값: {loss_sotr}")
        
    except Exception as e:
        print(f"forward 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

    print("\nDSFD_BRNet.py 실행 테스트 종료.")