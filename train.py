
import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import pyvista as pv

from dataset import VortexMAEDataset
from model import VortexMAE, vortex_mae_pretrain_loss
from vortex_utils import calculate_psnr, calculate_masked_psnr

def setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
        dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:23456", world_size=1, rank=0)
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def main():
    parser = argparse.ArgumentParser(description="VortexMAE Pre-training (AMP+Cosine Optimized)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8) # Total batch size
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_pretrain")
    args = parser.parse_args()
    
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"vortexmae2: Initializing Pre-training on {world_size} GPUs. Target LR: {args.lr}")

    train_dataset = VortexMAEDataset(args.data_dir, split="pretrain_train", augment=False)
    test_dataset = VortexMAEDataset(args.data_dir, split="pretrain_eval", augment=False)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=max(1, args.batch_size // world_size), num_workers=4)
    train_loader = DataLoader(train_dataset, batch_size=max(1, args.batch_size // world_size), sampler=train_sampler, num_workers=4)
    
    model = VortexMAE(in_chans=3, mask_ratio=args.mask_ratio, embed_dim=96, depths=[2, 2, 12, 2]).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler1 = LinearLR(optimizer, start_factor=0.01, total_iters=10)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=args.epochs - 10, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[10])
    scaler = GradScaler()
    
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        train_loss = torch.tensor(0.0).to(device)
        train_psnr = torch.tensor(0.0).to(device)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(rank != 0))
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            with autocast():
                x_rec, mask = model(batch)
                loss = vortex_mae_pretrain_loss(x_rec, batch, mask)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.detach()
            train_psnr += calculate_masked_psnr(x_rec, batch, mask).detach()
            
        # Synchronize across GPUs
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_psnr, op=dist.ReduceOp.SUM)
        
        avg_train_loss = train_loss.item() / (len(train_loader) * world_size)
        avg_train_psnr = train_psnr.item() / (len(train_loader) * world_size)
        
        # Validation every epoch
        model.eval()
        test_loss, test_psnr = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
        with torch.no_grad(), autocast():
            for batch in test_loader:
                batch = batch.to(device)
                x_rec, mask = model(batch)
                test_loss += vortex_mae_pretrain_loss(x_rec, batch, mask).detach()
                test_psnr += calculate_masked_psnr(x_rec, batch, mask).detach()
        
        dist.all_reduce(test_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(test_psnr, op=dist.ReduceOp.SUM)
        
        avg_test_loss = test_loss.item() / (len(test_loader) * world_size)
        avg_test_psnr = test_psnr.item() / (len(test_loader) * world_size)
        
        if rank == 0:
            print(f"Epoch {epoch} | Train MSE: {avg_train_loss:.6f} PSNR: {avg_train_psnr:.2f}dB | Test MSE: {avg_test_loss:.6f} PSNR: {avg_test_psnr:.2f}dB")
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                torch.save({'model_state_dict': model.module.state_dict()}, os.path.join(args.save_dir, "vortexmae_best.pth"))
        
        scheduler.step()
    
    if rank == 0: print("Pre-training Complete.")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
