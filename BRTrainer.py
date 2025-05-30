import os
import time
import torch
import wandb
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.amp import autocast
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim

from models.losses.multibox_loss import MultiBoxLoss
from models.losses.enhance_loss import EnhanceLoss
from utils.DarkISP import Low_Illumination_Degrading
from models.data.augmentations import Compute_Darklevel


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    # lr = args.lr * args.batch_size / 4 * torch.cuda.device_count() * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * gamma

# ============================== 
# Trainer Class
# ============================== 
        
class BR_Trainer:
    def __init__(self, 
                 model, 
                 net_enh,
                 train_loader,
                 val_loader,
                 optimizer,
                 cfg,
                 args,
                 epochs=100,
                 start_epoch = None,
                 eval_steps=500,
                 checkpoint_dir="./ckpt",
                 use_wandb=True,
                 project_name="BRNet"
                 ):
        
        self.cfg = cfg
        self.args = args
        self.model = model
        self.net_enh = net_enh
        self.device = "cuda" if args.cuda else "cpu"
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optim = optimizer
        # self.device = device # __init__ 인자에서 device를 받지 않고 args.cuda로 결정

        # 모델과 enhancer를 적절한 장치로 이동
        self.model.to(self.device)
        self.net_enh.to(self.device)
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.iteration = 0 if start_epoch is None else start_epoch * len(train_loader)
        self.step_index = 0


        self.eval_steps = eval_steps
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.use_wandb = use_wandb

        
        # define loss functions
        self.criterion = MultiBoxLoss(cfg, args.cuda)
        self.criterion_enhance = EnhanceLoss()
        self.criterion_dark = nn.MSELoss()

        # define for our new semi-orthogonal regularity loss
        self.ort_func = nn.CosineSimilarity(dim=1, eps=1e-8)

        self.grads = {}

    # define for gradient saving, (for S-ORT loss)
    def save_grad(self, name):
        def hook(grad):
            self.grads[name] = grad.detach()
        return hook

        


    def compute_losses(self, batch, mode = "train"):
        """
        return losses, time
        """



        with autocast('cuda', dtype=torch.bfloat16 if self.cfg.BF16 else torch.float32):
            images, targets, darklevels, img_paths = batch
            images = images.to(self.device) / 255.
            detection_targets = [cls.to(self.device) for cls in targets] # 변수명 명확화
            img_dark = torch.empty_like(images).to(self.device) 

            hook_dict = {
                'dark': self.save_grad('dark_grad'),
                'light': self.save_grad('light_grad'),
                }


            # Compute_Darklevel 함수는 (C,H,W) 형태의 단일 이미지를 입력으로 받음
            # DataLoader에서 오는 darklevels의 형태와 Compute_Darklevel의 사용 일관성 확인 필요
            
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
                dl_light_gt_list.append(darklevels[i].to(self.device) if isinstance(darklevels, list) else darklevels[i:i+1].to(self.device))
           
            self.iteration += 1
            if self.iteration in self.cfg.LR_STEPS:
                self.step_index += 1
                adjust_learning_rate(self.optim, self.args.gamma, self.step_index)

            if self.iteration >= self.cfg.MAX_STEPS:
                # MAX_STEPS 도달 시 None 반환하여 train_epoch에서 처리하도록 함
                return None, None, None

            dl_dark_gt = torch.stack(dl_dark_gt_list).view(-1, 1, 1, 1)
            dl_light_gt = torch.stack(dl_light_gt_list).view(-1, 1, 1, 1)

            t0 = time.time()
            R_dark_gt, I_dark = self.net_enh(img_dark)
            R_light_gt, I_light = self.net_enh(images)

            # self.model (DSFD_BRNet 인스턴스) 호출
            if mode == 'train':
                out, out2, out3, loss_mutual = self.model(img_dark, images, I_dark.detach(), I_light.detach(), hook_dict = hook_dict) # out2는 R_dark, R_light, R_dark_2, R_light_2
            else:
                out, out2, out3, loss_mutual = self.model(img_dark, images, I_dark.detach(), I_light.detach(), hook_dict = None)
            R_dark, R_light, R_dark_2, R_light_2 = out2
            dl_dark, dl_light = out3
            

            # backprop
            self.optim.zero_grad()

            loss_l_pal1, loss_c_pal1 = self.criterion(out[:3], detection_targets)
            loss_l_pal2, loss_c_pal2 = self.criterion(out[3:], detection_targets)

            loss_enhance = self.criterion_enhance([R_dark, R_light, R_dark_2, R_light_2, I_dark.detach(), I_light.detach()], images, img_dark) * 0.1
            loss_enhance2 = F.l1_loss(R_dark, R_dark_gt.detach()) + F.l1_loss(R_light, R_light_gt.detach()) + (
                        1. - ssim(R_dark, R_dark_gt.detach())) + (1. - ssim(R_light, R_light_gt.detach()))
            
            # darklevel loss
            loss_darklevel_dark = self.criterion_dark(dl_dark, dl_dark_gt) 
            loss_darklevel_light = self.criterion_dark(dl_light, dl_light_gt) 
            loss_darklevel = (loss_darklevel_dark + loss_darklevel_light ) * self.cfg.WEIGHT.DL

            if not self.cfg.ABLATION.SORT:
                loss_sort = torch.tensor(0.0).to(self.device)
            else:
                # hooking한 gradient 가져오기
                if mode != 'train':
                    g_light = self.grads.get('light_grad', None)
                    g_dark  = self.grads.get('dark_grad', None)

                    if g_light is not None and g_dark is not None:


                        g_light_flat = g_light.view(g_light.size(0), -1)
                        g_dark_flat  = g_dark.view(g_dark.size(0), -1)

                        sort_idx = int(g_light_flat.size(1) / self.cfg.WEIGHT.SORT_RATIO)

                        g_light_flat_part = g_light_flat[:, :sort_idx]
                        g_dark_flat_part = g_dark_flat[:, :sort_idx]

                        # ORT loss from "https://github.com/cuiziteng/ICCV_MAET.git" of https://arxiv.org/abs/2205.03346
                        loss_sort = self.cfg.WEIGHT.SORT * torch.mean(torch.abs(self.ort_func(g_light_flat_part, g_dark_flat_part)))\
                            + self.cfg.WEIGHT.SORT_M*torch.mean(1 - torch.abs(self.ort_func(g_light_flat_part, g_light_flat_part)))\
                            + self.cfg.WEIGHT.SORT_M*torch.mean(1 - torch.abs(self.ort_func(g_dark_flat_part, g_dark_flat_part)))

                    else:
                        loss_sort = torch.tensor(0.0).to(self.device)


                else:
                    loss_sort = torch.tensor(0.0).to(self.device)
                

            loss = loss_l_pal1 + loss_c_pal1 + loss_l_pal2 + loss_c_pal2 + loss_enhance2 + loss_enhance \
                + loss_darklevel + loss_mutual + loss_sort
        
            
            
            t1 = time.time()
            training_time = t1 - t0
            losses = {
                "loss_l_pal1": loss_l_pal1.item(),
                "loss_c_pal1": loss_c_pal1.item(),
                "loss_l_pal2": loss_l_pal2.item(),
                "loss_c_pal2": loss_c_pal2.item(),
                "loss_enhance": loss_enhance.item(),
                "loss_enhance2": loss_enhance2.item(),
                "loss_darklevel": loss_darklevel.item(),
                "loss_mutual": loss_mutual.item(),
                "loss_sort": loss_sort.item(),
                "total_loss": loss.item() # 키 이름 변경하여 명확성 확보
            }

        return loss, losses, training_time


    def train_epoch(self, epoch):
        if isinstance(self.train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            self.train_loader.sampler.set_epoch(epoch)

        self.model.train()
        total_loss= 0.0
        eval_loss = 0.0

        total_loss_l_pal1 = 0.0
        total_loss_c_pal1 = 0.0
        total_loss_l_pal2 = 0.0
        total_loss_c_pal2 = 0.0
        total_loss_enhance = 0.0
        total_loss_enhance2 = 0.0
        total_loss_darklevel = 0.0
        total_loss_mutual = 0.0
        total_loss_sort = 0.0

        epoch_bar = tqdm(self.train_loader, desc="Training Progress")
        for batch_idx, batch in enumerate(epoch_bar):

            if self.iteration >= self.cfg.MAX_STEPS:
                return False
            
            # self.model.train() # 에폭 시작 시 한 번만 호출
            model_loss, losses, _ = self.compute_losses(batch, mode = 'train')

            if model_loss is None: # MAX_STEPS 도달 시
                return False


            model_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Optimizer step 및 gradient 초기화
            self.optim.step()
            self.optim.zero_grad()


            total_loss += model_loss.item()

            # accumulate losses
            total_loss_l_pal1 += losses['loss_l_pal1']
            total_loss_c_pal1 += losses['loss_c_pal1']
            total_loss_l_pal2 += losses['loss_l_pal2']
            total_loss_c_pal2 += losses['loss_c_pal2']
            total_loss_enhance += losses['loss_enhance']
            total_loss_enhance2 += losses['loss_enhance2']
            total_loss_darklevel += losses['loss_darklevel']
            total_loss_mutual += losses['loss_mutual']
            total_loss_sort += losses['loss_sort']

            batch_num = batch_idx + 1



            # WandB 로깅 (10 iter마다)
            if self.use_wandb and (batch_idx % 1 == 0):
                log_data = {
                    "train/loss_l_pal1": total_loss_l_pal1/batch_num,
                    "train/loss_c_pal1": total_loss_c_pal1/batch_num,
                    "train/loss_l_pal2": total_loss_l_pal2/batch_num,
                    "train/loss_c_pal2": total_loss_c_pal2/batch_num,
                    "train/loss_enhance": total_loss_enhance/batch_num,
                    "train/loss_enhance2": total_loss_enhance2/batch_num,
                    "train/loss_darklevel": total_loss_darklevel/batch_num,
                    "train/loss_mutual": total_loss_mutual/batch_num,
                    "train/loss_sort": total_loss_sort/batch_num,
                    "epoch": epoch,
                    "step": epoch * len(self.train_loader) + batch_num
                }
                
                if self.args.multigpu:
                    if (self.args.local_rank == 0):
                        wandb.log(log_data)
                else:
                    wandb.log(log_data)



            #if batch_idx + 1 == len(self.train_loader):
            if batch_idx  == 0:
                eval_loss = self.evaluate()

                

            if epoch_bar is not None:
                epoch_bar.set_postfix({
                    "Train_Loss" : f"{total_loss/((batch_idx +1) * self.args.batch_size):.6f}", # self.args.batch_size 사용
                    "Eval_Loss" : f"{eval_loss/len(self.val_loader):.6f}",
                    "Epoch": epoch,
                    "Iteration": self.iteration,
                    "LR": f"{self.optim.param_groups[0]['lr']:.6f}",
                    })
            
                epoch_bar.update(1)
            
            break

        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        total_val_loss = 0.0

        total_val_loss_l_pal1 = 0.0
        total_val_loss_c_pal1 = 0.0
        total_val_loss_l_pal2 = 0.0
        total_val_loss_c_pal2 = 0.0
        total_val_loss_enhance = 0.0
        total_val_loss_enhance2 = 0.0
        total_val_loss_darklevel = 0.0
        total_val_loss_mutual = 0.0
        total_val_loss_sort = 0.0

        with torch.no_grad():
            cnt = 0 

            for batch in self.val_loader:
                model_loss, losses_dict, _ = self.compute_losses(batch, mode = 'val')
                if model_loss is None: # MAX_STEPS 도달 시 (평가 중에는 발생하지 않아야 하지만 안전장치)
                    continue

                total_val_loss += model_loss.item()
                cnt += 1
                
                total_val_loss_l_pal1 += losses_dict['loss_l_pal1']
                total_val_loss_c_pal1 += losses_dict['loss_c_pal1']
                total_val_loss_l_pal2 += losses_dict['loss_l_pal2']
                total_val_loss_c_pal2 += losses_dict['loss_c_pal2']
                total_val_loss_enhance += losses_dict['loss_enhance']
                total_val_loss_enhance2 += losses_dict['loss_enhance2']
                total_val_loss_darklevel += losses_dict['loss_darklevel']
                total_val_loss_mutual += losses_dict['loss_mutual']
                total_val_loss_sort += losses_dict['loss_sort']

                

        if self.use_wandb:
            log_data = {
                "val/loss_l_pal1": total_val_loss_l_pal1 / cnt,
                "val/loss_c_pal1": total_val_loss_c_pal1 / cnt,
                "val/loss_l_pal2": total_val_loss_l_pal2 / cnt,
                "val/loss_c_pal2": total_val_loss_c_pal2 / cnt,
                "val/loss_enhance": total_val_loss_enhance / cnt,
                "val/loss_enhance2": total_val_loss_enhance2 / cnt,
                "val/loss_darklevel": total_val_loss_darklevel / cnt,
                "val/loss_mutual": total_val_loss_mutual / cnt,
                "val/loss_sort": total_val_loss_sort / cnt,
                "val/total_loss": total_val_loss / cnt
            }
        
            if self.args.multigpu:
                if (self.args.local_rank == 0):
                    wandb.log(log_data)
            else:
                wandb.log(log_data)

        return total_val_loss / cnt if cnt > 0 else 0.0 # 평균 손실 반환

    def save_checkpoint(self, epoch, is_best=False):
        """
        모델과 옵티마이저 상태를 저장

        Args:
            epoch (int): 현재 에폭
            is_best (bool): 성능이 가장 좋을 때만 저장하고 싶을 경우 사용
        """
        # get model without DataParallel or DDP wrapper
        core_model = self.get_core_model()
        
        # 저장 경로 생성
        save_name = f"{self.cfg.MODEL_NAME}.pt"
        save_path = os.path.join(self.checkpoint_dir, save_name)

        # 체크포인트 딕셔너리 구성
        checkpoint = {
            "model": core_model.state_dict(),
            "optimizer": self.optim.state_dict(),
            "epoch": epoch
        }

        # 저장 (모델은 GPU에 있어도 무방, 파일은 CPU로 저장됨)
        torch.save(checkpoint, save_path)

        # 성능이 가장 좋을 때 저장하는 별도 버전
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f"{self.cfg.MODEL_NAME}_best.pt")
            torch.save(checkpoint, best_path)


    def get_core_model(self):
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self.model.module
        return self.model

    def train(self):
        self.best_loss = torch.inf  # 초기 최저 손실값 설정
        for epoch in range(self.epochs):
            avg_loss = self.train_epoch(epoch)

            
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.save_checkpoint(epoch)

            """
            if not avg_loss : #train_epoch 메소드가 False를 리턴하면 종료
                print("Current Step exeeds the Maximum Steps")
                break
            """

        # 학습 완료 후 최종 모델 저장 시에도 CPU로 옮겨서 저장
        if self.args.local_rank == 0: # 메인 프로세스에서만 저장

            final_model_state_dict = self.get_core_model().to('cpu').state_dict()
            torch.save(final_model_state_dict, os.path.join(self.checkpoint_dir, f"{self.cfg.MODEL_NAME}_final.pt"))
            if isinstance(self.model, nn.DataParallel) or isinstance(self.model, nn.parallel.DistributedDataParallel):
                self.model.module.to(self.device)
            else:
                self.model.to(self.device)



if __name__ == "__main__":
    pass
