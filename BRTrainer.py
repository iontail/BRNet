import os
import time
import torch
import wandb
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import L1Loss, CrossEntropyLoss
from torch.amp import autocast
from src.models.our_model import Base_Model
import numpy as np # Added for main block

# from src.utils.metrics import compute_metrics  # 필요시 사용

class BR_Trainer:
    def __init__(self, 
                 model, 
                 slm,
                 train_loader,
                 val_loader,
                 optimizer,
                 device,
                 cfg,
                 epochs=100,
                 eval_steps=500,
                 checkpoint_dir="./ckpt",
                 use_wandb=True,
                 project_name="MB_SLM",
                 run_name=None
                 ):
        
        self.cfg = cfg
        self.model = model
        self.slm = slm

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optim = optimizer
        self.device = device

        self.epochs = epochs
        self.eval_steps = eval_steps
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.use_wandb = use_wandb


        # define loss functions
        self.ce_loss = CrossEntropyLoss()


    def compute_losses(self, batch):

        # 데이터 로딩: UTKFaceDataset은 (img_tensor, label_tensor) 튜플을 반환
        img, labels = batch
        img = img.to(self.device)
        labels = labels.to(self.device)

        # 설정 파일(cfg)에서 지정한 인덱스로 주요 타겟 레이블 선택
        target = labels[:, self.cfg.DATASET.TARGET_LABEL_IDX]
        with autocast('cuda', dtype=torch.bfloat16 if self.cfg.BF16 else torch.float32):
            # generator forward pass 한 번 실행
            logits, features = self.model(img)

            # SLM 통과하는 부분 추가해줘야 할듯
            ce_loss = self.ce_loss(logits, target)

            # R1 정규화: 실제 이미지에 대한 gradient 계산
            """
            if train:
                img_tmp = img.detach().requires_grad_(True)
                r1_logits = self.model(img_tmp)
                r1_grads = torch.autograd.grad(outputs=r1_logits.sum(), inputs=img_tmp, create_graph=True)[0]
                r1_penalty = r1_grads.pow(2).sum([1, 2, 3]).mean()
                r1_loss = r1_penalty * (self.cfg.R1_GAMMMA / 2)
            else:
                r1_loss = torch.tensor(0.0).to(img.device)

            model_loss = ce_loss + self.cfg.R1_LAMBDA * r1_loss # .item() 제거 - backward를 위해 텐서 유지
            """
            r1_loss = torch.tensor(0.0).to(img.device) # R1 loss는 현재 사용하지 않음
            model_loss = ce_loss + self.cfg.R1_LAMBDA * r1_loss

        losses = {
            "ce_loss": ce_loss,
            'r1_loss': r1_loss,
            "loss": model_loss
        }

        

        #return model_loss, losses, outputs
        return model_loss, losses, logits


    def train_epoch(self, epoch):
        self.model.train()

        total_loss= 0.0
        eval_loss = 0.0


        #for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}")):
        epoch_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", unit="epoch", position=0, leave=True)
        for batch_idx, batch in enumerate(epoch_bar):
            # 한 배치당 forward pass 1회로 두 손실을 모두 계산
            # compute_losses 내부에서 autocast를 처리하므로 여기서는 제거
            self.model.train()
            model_loss, losses, outputs = self.compute_losses(batch)


            self.model.requires_grad_(True)
            model_loss.backward() # retain_graph=True 불필요 (한 번의 forward로 계산)


            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Optimizer step 및 gradient 초기화
            self.optim.step()
            self.optim.zero_grad()


            total_loss += model_loss.item()


            # WandB 로깅 (10 iter마다)
            if self.use_wandb and (batch_idx % 10 == 0):
                wandb.log({
                    "train/batch_loss": model_loss.item(), # 현재 배치의 손실 기록
                    "train/r1_loss": losses['r1_loss'].item(),
                    "train/ce_loss": losses['ce_loss'].item(),
                    "epoch": epoch,
                    "step": epoch * len(self.train_loader) + batch_idx
                })


            if batch_idx + 1 == len(self.train_loader):
                eval_loss = self.evaluate()
            epoch_bar.set_postfix(train_loss=f"{total_loss/((batch_idx +1) * self.cfg.BATCH_SIZE):.6f}", eval_loss=f"{eval_loss/len(self.val_loader):.6f}")


        

        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0

        log_images = []
        with torch.no_grad():
            cnt = 0
            for batch in self.val_loader:
                model_loss, losses, outputs = self.compute_losses(batch)
                total_loss += model_loss.item()
                cnt += 1

        if self.use_wandb:
            wandb.log({
                "eval/avg_loss": total_loss / cnt, # Use average loss
                "eval/r1_loss": losses['r1_loss'].item(), # Note: losses from the *last* batch
                "eval/ce_loss": losses['ce_loss'].item(),
            })
        if self.use_wandb and log_images:
            wandb.log({
                "eval/images": [wandb.Image(img) for img in log_images]
            })


        return total_loss

    def save_checkpoint(self, epoch):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optim.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"{self.cfg.MODEL_NAME}.pt"))

    def train(self):
        for epoch in range(self.epochs):
            start_time = time.time()
            avg_loss = self.train_epoch(epoch)
            if self.use_wandb:
                wandb.log({
                    "epoch/avg_loss": avg_loss,
                    "epoch/time": time.time() - start_time
                })
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1)

        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f"{cfg.MODEL_NAME}.pt"))


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
