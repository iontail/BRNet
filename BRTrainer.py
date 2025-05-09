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
                 project_name="Bio_Reflect_Net",
                 run_name=None
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

        


    def compute_losses(self, batch):
        """
        return losses, time
        """

        with autocast('cuda', dtype=torch.bfloat16 if self.cfg.BF16 else torch.float32):
            images, targets, darklevels, img_paths = batch
            images = images.to(self.device) / 255.
            detection_targets = [ann.to(self.device) for ann in targets] # 변수명 명확화
            img_dark = torch.empty_like(images).cuda()

            # darklevels는 WIDERDetection에서 로드된 값 (아마도 원본 이미지에서 계산된 값)
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
                # WIDERDetection에서 오는 darklevels가 이미 계산된 값이라면 그것을 사용
                # 아니라면 여기서 계산
                dl_dark_gt_list.append(Compute_Darklevel(img_dark[i].cpu()).to(self.device)) # CPU에서 계산 후 GPU로
                dl_light_gt_list.append(darklevels[i].to(self.device) if isinstance(darklevels, list) else darklevels[i:i+1].to(self.device)) # 데이터로더 출력에 따라 수정
           
            self.iteration += 1
            if self.iteration in self.cfg.LR_STEPS:
                self.step_index += 1
                adjust_learning_rate(self.optim, self.args.gamma, self.step_index) # self.optim 사용

            if self.iteration >= self.cfg.MAX_STEPS:
                # MAX_STEPS 도달 시 None 반환하여 train_epoch에서 처리하도록 함
                return None, None, None

            dl_dark_gt = torch.stack(dl_dark_gt_list).unsqueeze(1).unsqueeze(2).unsqueeze(3) # (B, 1, 1, 1) 형태로
            dl_light_gt = torch.stack(dl_light_gt_list).unsqueeze(1).unsqueeze(2).unsqueeze(3) # (B, 1, 1, 1) 형태로

            t0 = time.time()
            R_dark_gt, I_dark = self.net_enh(img_dark)
            R_light_gt, I_light = self.net_enh(images)

            # self.model (DSFD_BRNet 인스턴스) 호출
            out, out2, out3, loss_mutual, loss_sotr = self.model(img_dark, images, I_dark.detach(), I_light.detach())
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
            loss_darklevel_dark = self.criterion_dark(dl_dark, dl_dark_gt) # dl_dark_gt 사용
            loss_darklevel_light = self.criterion_dark(dl_light, dl_light_gt) # dl_light_gt 사용
            loss_darklevel = (loss_darklevel_dark + loss_darklevel_light ) * self.cfg.WEIGHT.DL


            loss = loss_l_pal1 + loss_c_pal1 + loss_l_pal2 + loss_c_pal2 + loss_enhance2 + loss_enhance \
                + loss_darklevel + loss_mutual + loss_sotr
            
            
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
                "loss_sotr": loss_sotr.item(),
                "total_loss": loss.item() # 키 이름 변경하여 명확성 확보
            }

        return loss, losses, training_time


    def train_epoch(self, epoch):
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
        total_loss_sotr = 0.0

        for batch_idx, batch in enumerate(self.train_loader):

            if self.iteration >= self.cfg.MAX_STEPS:
                return False
            
            # self.model.train() # 에폭 시작 시 한 번만 호출
            model_loss, losses, _ = self.compute_losses(batch)

            if model_loss is None: # MAX_STEPS 도달 시
                return False


            model_loss.backward() # retain_graph=True 불필요 (한 번의 forward로 계산)

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
            total_loss_sotr += losses['loss_sotr']

            batch_num = batch_idx + 1



            # WandB 로깅 (10 iter마다)
            if self.use_wandb and (batch_idx % 10 == 0 and batch_idx != 0):
                wandb.log({
                    "train/loss_l_pal1": total_loss_l_pal1/batch_num,
                    "train/loss_c_pal1": total_loss_c_pal1/batch_num,
                    "train/loss_l_pal2": total_loss_l_pal2/batch_num,
                    "train/loss_c_pal2": total_loss_c_pal2/batch_num,
                    "train/loss_enhance": total_loss_enhance/batch_num,
                    "train/loss_enhance2": total_loss_enhance2/batch_num,
                    "train/loss_darklevel": total_loss_darklevel/batch_num,
                    "train/loss_mutual": total_loss_mutual/batch_num,
                    "train/loss_sotr": total_loss_sotr/batch_num,
                    "epoch": epoch,
                    "step": epoch * len(self.train_loader) + batch_num
                })


            if batch_idx + 1 == len(self.train_loader):
                eval_loss = self.evaluate()

            self.epoch_bar.set_postfix({
                "Train_Loss" : f"{total_loss/((batch_idx +1) * self.args.batch_size):.6f}", # self.args.batch_size 사용
                "Eval_Loss" : f"{eval_loss/len(self.val_loader):.6f}",
                "Epoch": epoch,
                "Iteration": self.iteration,
                "LR": f"{self.optim.param_groups[0]['lr']:.6f}",
                })
            
            self.epoch_bar.update(1)

        

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
        total_val_loss_sotr = 0.0

        with torch.no_grad():
            cnt = 0 

            for batch in self.val_loader:
                model_loss, losses_dict, _ = self.compute_losses(batch)
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
                total_val_loss_sotr += losses_dict['loss_sotr']

                

        if self.use_wandb:
            wandb.log({
                "val/loss_l_pal1": total_val_loss_l_pal1 / cnt,
                "val/loss_c_pal1": total_val_loss_c_pal1 / cnt,
                "val/loss_l_pal2": total_val_loss_l_pal2 / cnt,
                "val/loss_c_pal2": total_val_loss_c_pal2 / cnt,
                "val/loss_enhance": total_val_loss_enhance / cnt,
                "val/loss_enhance2": total_val_loss_enhance2 / cnt,
                "val/loss_darklevel": total_val_loss_darklevel / cnt,
                "val/loss_mutual": total_val_loss_mutual / cnt,
                "val/loss_sotr": total_val_loss_sotr / cnt,
                "val/total_loss": total_val_loss / cnt
            })

        return total_val_loss / cnt if cnt > 0 else 0.0 # 평균 손실 반환

    def save_checkpoint(self, epoch):
        # 모델을 CPU로 옮겨서 state_dict를 가져옵니다.
        model_state_dict = self.model.module.to('cpu').state_dict() if isinstance(self.model, nn.DataParallel) or isinstance(self.model, nn.parallel.DistributedDataParallel)\
            else self.model.to('cpu').state_dict()
        
        checkpoint = {
            "model": model_state_dict,
            "optimizer": self.optim.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"{self.cfg.MODEL_NAME}.pt"))
        
        # 모델을 원래 device로 다시 옮기기 (학습 계속하기 위해)
        if isinstance(self.model, nn.DataParallel) or isinstance(self.model, nn.parallel.DistributedDataParallel):
            self.model.module.to(self.device) # DataParallel/DDP의 경우 원래 모듈을 옮김
        else:
            self.model.to(self.device)

    def train(self):
        self.net_enh.eval()

        self.epoch_bar = tqdm(total=self.epochs, desc="Training Progress")
        for epoch in self.epoch_bar:
            avg_loss = self.train_epoch(epoch)

            
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1)

            if not avg_loss: #train_epoch 메소드가 False를 리턴하면 종료
                print("Current Step exeeds the Maximum Steps")
                break

        # 학습 완료 후 최종 모델 저장 시에도 CPU로 옮겨서 저장
        if self.args.local_rank == 0: # 메인 프로세스에서만 저장
            final_model_state_dict = self.model.module.to('cpu').state_dict() if isinstance(self.model, nn.DataParallel) or isinstance(self.model, nn.parallel.DistributedDataParallel) \
                else self.model.to('cpu').state_dict()
            torch.save(final_model_state_dict, os.path.join(self.checkpoint_dir, f"{self.cfg.MODEL_NAME}_final.pt"))
            if isinstance(self.model, nn.DataParallel) or isinstance(self.model, nn.parallel.DistributedDataParallel):
                self.model.module.to(self.device)
            else:
                self.model.to(self.device)



if __name__ == "__main__":
    from models.utils.config import cfg
    from datasets.dataset_UTKFace import UTKFace_Dataset, augment_train, augment_test
    import os # For path joining


    # --- Configuration ---
    if not os.path.exists(cfg.DATASET.ROOT_DIR):
         raise FileNotFoundError(f"UTKFace dataset not found at {cfg.DATASET.ROOT_DIR}. Please set cfg.DATASET.ROOT_DIR correctly.")

    # --- Dataset & DataLoader ---
    train_ds = UTKFace_Dataset(
        root_dir=cfg.DATASET.ROOT_DIR,
        train_mode=True,
        transform=augment_train, # Use train transforms
        test_split_ratio=cfg.DATASET.TEST_SPLIT_RATIO,
        random_state=cfg.DATASET.RANDOM_STATE
    )
    val_ds = UTKFace_Dataset(
        root_dir=cfg.DATASET.ROOT_DIR,
        train_mode=False,
        transform=augment_test, # Use test transforms
        test_split_ratio=cfg.DATASET.TEST_SPLIT_RATIO,
        random_state=cfg.DATASET.RANDOM_STATE
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.DATASET.NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.DATASET.NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = Base_Model(cfg=cfg).to(device) # Ensure Base_Model's output matches cfg.MODEL.NUM_CLASSES
    optim  = Adam(model.parameters(), lr=1e-3)


    trainer = MB_SLMTrainer(
        model         = model,
        slm           = None,           # 아직 안 쓰는 경우 None
        train_loader  = train_loader,
        val_loader    = val_loader,
        optimizer     = optim,
        device        = device,
        cfg           = cfg,
        epochs        = 2, # Increase epochs for real dataset
        eval_steps    = 200,
        checkpoint_dir= "./ckpt",
        use_wandb     = True           
    )

    trainer.train()
