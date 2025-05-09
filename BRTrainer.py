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

# from src.utils.metrics import compute_metrics  # 필요시 사용

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
        self.device = device

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
        self.criterion_dark = nn.BCELoss()

        for i in range(2):
            if self.cfg.LR_STEPS[i] <= self.iteration < self.cfg.LR_STEPS[i+1]:
                self.step_index = i
                break


    def compute_losses(self, batch):
        """
        return losses, time
        """

        with autocast('cuda', dtype=torch.bfloat16 if self.cfg.BF16 else torch.float32):
            images, targets, darklevels, img_paths = batch
            images = images.to(self.device) / 255.
            targetss = [ann.to(self.device) for ann in targets]
            img_dark = torch.empty_like(images).cuda()

            dl_light_targets = [darklevel.to(self.device) for darklevel in darklevels]
            dl_dark_targets = []

            dark_target = []
            dark_light_target = []

            # Generation of degraded data and AET groundtruth
            for i in range(images.shape[0]):
                img_dark[i], _ = Low_Illumination_Degrading(images[i])

                # computing dark-level labeling
                dl_dark_targets.append(Compute_Darklevel(img_dark[i]))

           
            self.iteration += 1
            if self.iteration in self.cfg.LR_STEPS:
                self.step_index += 1
                adjust_learning_rate(self.optimizer, self.args.gamma, self.step_index)


            dark_target = torch.stack(dark_target, dim=0).cuda()
            dark_light_target = torch.stack(dark_light_target, dim=0).cuda()
            

            t0 = time.time()
            R_dark_gt, I_dark = self.net_enh(img_dark)
            R_light_gt, I_light = self.net_enh(images)

            out, out2, out3, loss_mutual, loss_sotr = self.net(img_dark, images, I_dark.detach(), I_light.detach())
            R_dark, R_light, R_dark_2, R_light_2 = out2
            dl_dark, dl_light = out3
            

            # backprop
            self.optim.zero_grad()

            loss_l_pal1, loss_c_pal1 = self.criterion(out[:3], targetss)
            loss_l_pal2, loss_c_pal2 = self.criterion(out[3:], targetss)

            loss_enhance = self.criterion_enhance([R_dark, R_light, R_dark_2, R_light_2, I_dark.detach(), I_light.detach()], images, img_dark) * 0.1
            loss_enhance2 = F.l1_loss(R_dark, R_dark_gt.detach()) + F.l1_loss(R_light, R_light_gt.detach()) + (
                        1. - ssim(R_dark, R_dark_gt.detach())) + (1. - ssim(R_light, R_light_gt.detach()))
            
            # darklevel loss
            loss_darklevel_dark = self.criterion_dark(dl_dark, dl_dark_targets)
            loss_darklevel_light = self.criterion_dark(dl_light, dl_light_targets)
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
                "loss": loss.item()
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


        #for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}")):
        epoch_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", unit="epoch", position=0, leave=True)
        for batch_idx, batch in enumerate(epoch_bar):
            
            self.model.train()
            model_loss, losses, _ = self.compute_losses(batch)


            self.model.requires_grad_(True)
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
            if self.use_wandb and (batch_idx % 10 == 0):
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


    def train_epoch(self, epoch):
            for epoch in range(self.start_, cfg.EPOCHES):
                    losses = 0

                    pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{cfg.EPOCHES}") #

                    for batch_idx, (images, targets, _) in enumerate(train_loader):
                        


                        """
                        if iteration % 100 == 0:
                            tloss = losses / (batch_idx + 1)
                            if local_rank == 0:
                                print('Timer: %.4f' % (t1 - t0))
                                print('epoch:' + repr(epoch) + ' || iter:' +
                                    repr(iteration) + ' || Loss:%.4f' % (tloss))
                                print('->> pal1 conf loss:{:.4f} || pal1 loc loss:{:.4f}'.format(
                                    loss_c_pal1.item(), loss_l_pa1l.item()))
                                print('->> pal2 conf loss:{:.4f} || pal2 loc loss:{:.4f}'.format(
                                    loss_c_pal2.item(), loss_l_pa12.item()))
                                print('->>lr:{}'.format(optimizer.param_groups[0]['lr']))
                        """
                        # 진행 상황을 실시간으로 업데이트
                        pbar.set_postfix({
                            "Loss": f"{losses / (batch_idx + 1):.4f}", 
                            "Iter": iteration,
                            "LR": f"{optimizer.param_groups[0]['lr']:.6f}"
                        })
                        pbar.update(1)

                        if iteration != 0 and iteration % 5000 == 0:
                            if local_rank == 0:
                                print('Saving state, iter:', iteration)
                                file = 'dsfd_' + repr(iteration) + '.pth'
                                torch.save(dsfd_net.state_dict(),
                                        os.path.join(save_folder, file))
                        iteration += 1


                        if iteration != 0 and iteration % 10 == 0 and args.use_wandb:
                            wandb.log({
                                "train/loc_pal1": loss_l_pa1l.item(),
                                "train/conf_pal1": loss_c_pal1.item(),
                                "train/loc_pal2": loss_l_pa12.item(),
                                "train/conf_pal2": loss_c_pal2.item(),
                                "train/overall_loss": loss.item(),
                                "train/epoch": epoch,
                                "train/step": iteration,

                            })
                    # if local_rank == 0:
                    if (epoch + 1) >= 0:
                        val(epoch, net, dsfd_net, net_enh, criterion)
                    if iteration >= cfg.MAX_STEPS:
                        break


        def val(epoch, net, dsfd_net, net_enh, criterion):
            net.eval()
            step = 0
            losses = torch.tensor(0.).cuda()
            losses_enh = torch.tensor(0.).cuda()
            t1 = time.time()

            for batch_idx, (images, targets, img_paths) in enumerate(val_loader):
                if args.cuda:
                    images = Variable(images.cuda() / 255.)
                    targets = [Variable(ann.cuda(), volatile=True)
                            for ann in targets]
                else:
                    images = Variable(images / 255.)
                    targets = [Variable(ann, volatile=True) for ann in targets]
                img_dark = torch.stack([Low_Illumination_Degrading(images[i])[0] for i in range(images.shape[0])],
                                    dim=0)
                out, R = net.module.test_forward(img_dark)

                loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targets)
                loss_l_pa12, loss_c_pal2 = criterion(out[3:], targets)
                loss = loss_l_pa12 + loss_c_pal2

                losses += loss.item()
                step += 1
            dist.reduce(losses, 0, op=dist.ReduceOp.SUM)

            tloss = losses / step / torch.cuda.device_count()
            t2 = time.time()
            if local_rank == 0:
                print('Timer: %.4f' % (t2 - t1))
                print('test epoch:' + repr(epoch) + ' || Loss:%.4f' % (tloss))

            global min_loss
            if tloss < min_loss:
                if local_rank == 0:
                    print('Saving best state,epoch', epoch)
                    torch.save(dsfd_net.state_dict(), os.path.join(
                        save_folder, 'dsfd.pth'))
                min_loss = tloss

            if args.use_wandb:
                        wandb.log({
                            "val/loc_pal1": loss_l_pa1l.item(),
                            "val/conf_pal1": loss_c_pal1.item(),
                            "val/loc_pal2": loss_l_pa12.item(),
                            "val/conf_pal2": loss_c_pal2.item(),
                            "val/overall_loss": losses.item(),
                            "val/total_loss": tloss,
                        })

            states = {
                'epoch': epoch,
                'weight': dsfd_net.state_dict(),
            }
            if local_rank == 0:
                torch.save(states, os.path.join(save_folder, 'dsfd_checkpoint.pth'))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    # lr = args.lr * args.batch_size / 4 * torch.cuda.device_count() * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * gamma

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
