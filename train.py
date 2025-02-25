import os
import warnings
warnings.filterwarnings("ignore")

import torch
import random
import numpy as np
import wandb

from arguments import get_arguments
from src.datasets.get_dataset import get_dataset
from src.model.get_model import get_model
from src.CoReaPTrainer import CoReaPTrainer


def set_seed(seed):
    """랜덤 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    # 장치 설정 (GPU가 있으면 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    # 데이터셋 로드
    train_dataset, val_dataset, data_collator = get_dataset(args)

    # 데이터 로더 설정
    num_workers = min(4, os.cpu_count())  # CPU 코어 기반 워커 설정
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.per_device_train_batch_size, 
        shuffle=True,  # DDP 제거 -> shuffle 가능
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.per_device_eval_batch_size, 
        shuffle=False,  # 검증 데이터는 순서 유지
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True
    )

    # 모델 생성 및 장치 이동
    generator, discriminator = get_model(args)
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # 혼합 정밀도 변환 (bf16 사용 시)
    if args.bf16:
        generator = generator.to(torch.bfloat16)
        discriminator = discriminator.to(torch.bfloat16)

    # 옵티마이저 설정
    g_optimizer = torch.optim.Adam(
        generator.parameters(), 
        lr=args.learning_rate, 
        betas=(0.5, 0.999),
        foreach=True
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), 
        lr=args.learning_rate, 
        betas=(0.5, 0.999),
        foreach=True
    )

    # GPU 메모리 최적화
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cuda.cufft_plan_cache.clear()

    # WandB 초기화
    wandb.init(project='CoReaP', name=args.save_dir, config=args)

    # model load
    if args.load_model != None:
        print(f"Load model from {args.load_model}")
        checkpoint = torch.load(os.path.join(args.load_model))
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer'])


    # 트레이너 설정
    trainer = CoReaPTrainer(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        device=device,
        r1_gamma=args.r1_gamma,
        pcp_ratio=args.pcp_ratio,
        l1_ratio=args.l1_ratio,
        high_freq_ratio=args.high_freq_ratio,
        epochs=args.num_train_epochs,
        gradient_accumulation=args.gradient_accumulation_steps,
        eval_steps=args.eval_steps,
        checkpoint_dir=f"./ckpt/{args.save_dir}",
        use_wandb=True,
        project_name="CoReaP",
        run_name=args.save_dir,
        args=args
    )

    # 학습 실행
    trainer.train()


if __name__ == "__main__":
    args = get_arguments()
    main(args)