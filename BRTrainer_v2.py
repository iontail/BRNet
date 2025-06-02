from torch import nn
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.trainer import Trainer
from torch import nn
import torch
import numpy as np
import wandb
from utils.DarkISP import Low_Illumination_Degrading
from models.data.augmentations import Compute_Darklevel
from models.losses.multibox_loss import MultiBoxLoss
from models.losses.enhance_loss import EnhanceLoss
from models.data.config import cfg
from torchmetrics.functional import structural_similarity_index_measure as ssim
import torch.nn.functional as F

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    # lr = args.lr * args.batch_size / 4 * torch.cuda.device_count() * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * gamma

class BR_Trainer_v2(Trainer):
    def __init__(self, net_enh=None, custom_optimizer=None, use_wandb = False, **kwds):
        super().__init__(**kwds)
        self.net_enh = net_enh
        self.custom_optimizer = custom_optimizer
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_wandb = use_wandb

        self.criterion = MultiBoxLoss(cfg, self.device)
        self.criterion_enhance = EnhanceLoss()
        self.criterion_dark = nn.MSELoss()

        # define for our new semi-orthogonal regularity loss
        self.ort_func = nn.CosineSimilarity(dim=1, eps=1e-8)
        
        

    def create_optimizer(self):
        if self.custom_optimizer is not None:
            self.optimizer = self.custom_optimizer
            return self.optimizer
        else:
            return super().create_optimizer()
        
    def compute_loss(self, model, inputs, num_items_in_batch = None, return_outputs=False):
        image = inputs['images']
        label = inputs['targets']
        darklevel = inputs['darklevels']
        img_path = inputs['paths']
        
        
        image = image.to(self.device) / 255.0
        img_dark = torch.empty_like(image).to(self.device)
        detection_targets = [cls.to(self.device) for cls in label]

        # Compute loss
        mode = "train" if model.training else "eval"
        output, loss = self.get_loss_fn(model, self.net_enh, image, img_dark, darklevel, detection_targets, mode = mode)

        if return_outputs:
            return loss, output, label
        return loss
    
    def get_loss_fn(self,
                    model: nn.Module,
                    net_enh: nn.Module,
                    images: torch.Tensor,
                    img_dark: torch.Tensor,
                    darklevel: List[torch.Tensor],
                    label: List[torch.Tensor],
                    mode: str = 'train',
                    ):
        
        # dl_light_gt와 dl_dark_gt를 생성
        dl_light_gt_list = []
        dl_dark_gt_list = []

        # Generation of degraded data and AET groundtruth
        for i in range(images.shape[0]):
            img_dark[i], _ = Low_Illumination_Degrading(images[i])
            # computing dark-level labeling
            # Compute_Darklevel은 (C,H,W) 텐서를 받아 스칼라 텐서를 반환해야 함

            # img_dark[i]는 0-1 범위이므로, Compute_Darklevel이 0-255 범위로 스케일링
            dl_dark_gt_list.append(Compute_Darklevel((img_dark[i].detach() * 255.0).cpu()).to(self.device)) # CPU에서 계산 후 GPU로

            # 정상이미지는 WiderFace 데이터셋에서 계산한 darklevel을 사용
            dl_light_gt_list.append(darklevel[i].to(self.device) if isinstance(darklevel, list) else darklevel[i:i+1].to(self.device))
        

        dl_dark_gt = torch.stack(dl_dark_gt_list).view(-1, 1, 1, 1)
        dl_light_gt = torch.stack(dl_light_gt_list).view(-1, 1, 1, 1)

        R_dark_gt, I_dark = self.net_enh(img_dark)
        R_light_gt, I_light = self.net_enh(images)

        # self.model (DSFD_BRNet 인스턴스) 호출
        out, out2, out3, loss_mutual, loss_sort = self.model(img_dark, images, I_dark.detach(), I_light.detach()) # out2는 R_dark, R_light, R_dark_2, R_light_2

        R_dark, R_light, R_dark_2, R_light_2 = out2
        dl_dark, dl_light = out3
        

        loss_l_pal1, loss_c_pal1 = self.criterion(out[:3], label)
        loss_l_pal2, loss_c_pal2 = self.criterion(out[3:], label)

        loss_enhance = self.criterion_enhance([R_dark, R_light, R_dark_2, R_light_2, I_dark.detach(), I_light.detach()], images, img_dark) * 0.1
        loss_enhance2 = F.l1_loss(R_dark, R_dark_gt.detach()) + F.l1_loss(R_light, R_light_gt.detach()) + (
                    1. - ssim(R_dark, R_dark_gt.detach())) + (1. - ssim(R_light, R_light_gt.detach()))
        
        # darklevel loss
        loss_darklevel_dark = self.criterion_dark(dl_dark, dl_dark_gt) 
        loss_darklevel_light = self.criterion_dark(dl_light, dl_light_gt) 
        loss_darklevel = (loss_darklevel_dark + loss_darklevel_light ) * self.cfg.WEIGHT.DL

    

        if self.use_wandb:
            if mode == "train":
                wandb.log({
                    'train/loss_l_pal1': loss_l_pal1.item(),
                    'train/loss_c_pal1': loss_c_pal1.item(),
                    'train/loss_l_pal2': loss_l_pal2.item(),
                    'train/loss_c_pal2': loss_c_pal2.item(),
                    'train/loss_enhance': loss_enhance.item(),
                    'train/loss_enhance2': loss_enhance2.item(),
                    'train/loss_darklevel': loss_darklevel.item(),
                    'train/loss_mutual': loss_mutual.item(),
                    'train/loss_sort': loss_sort.item(),
                })
            else:
                wandb.log({
                    'val/loss_l_pal1': loss_l_pal1.item(),
                    'val/loss_c_pal1': loss_c_pal1.item(),
                    'val/loss_l_pal2': loss_l_pal2.item(),
                    'val/loss_c_pal2': loss_c_pal2.item(),
                    'val/loss_enhance': loss_enhance.item(),
                    'val/loss_enhance2': loss_enhance2.item(),
                    'val/loss_darklevel': loss_darklevel.item(),
                    'val/loss_mutual': loss_mutual.item(),
                    'val/loss_sort': loss_sort.item(),
                })


            

        loss = loss_l_pal1 + loss_c_pal1 + loss_l_pal2 + loss_c_pal2 + loss_enhance2 + loss_enhance \
            + loss_darklevel + loss_mutual + loss_sort
        
        return out, loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        model.eval()
        
        with torch.no_grad():
            eval_loss, pred, label = self.compute_loss(model,inputs,return_outputs = True)
        
        return (eval_loss,pred,label)