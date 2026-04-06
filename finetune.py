
import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

from dataset import VortexMAEDataset
from model import VortexMAE
from vortex_utils import vortex_mae_paper_loss, calculate_iou, calculate_ivd

def setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world_size = 0, 1
        dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:23457", world_size=1, rank=0)
        torch.cuda.set_device(0)
    return rank, world_size

def main():
    parser = argparse.ArgumentParser(description="VortexMAE Fine-tuning (AMP+Cosine Optimized)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--pretrained_ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-4) # Slightly lower for finetune
    parser.add_argument("--save_dir", type=str, default="./checkpoints_finetune")
    args = parser.parse_args()
    
    rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"vortexmae2: Initializing Fine-tuning on {world_size} GPUs. Pretrained: {args.pretrained_ckpt}")

    train_dataset = VortexMAEDataset(args.data_dir, split="finetune_train", augment=False)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=max(1, args.batch_size // world_size), sampler=train_sampler, num_workers=4)
    
    model = VortexMAE(in_chans=3, out_chans=1, mode='segmentation', embed_dim=96, depths=[2, 2, 12, 2]).to(device)
    
    # Load pre-trained encoder weights
    checkpoint = torch.load(args.pretrained_ckpt, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'): k = k[7:]
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()
    
    best_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_loss, epoch_iou = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(rank != 0))
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            with autocast():
                # Ground Truth calculation on the fly
                with torch.no_grad():
                    gt_ivd = calculate_ivd(batch)
                    gt_mask = (gt_ivd > 0).float().unsqueeze(1)
                
                pred_logits = model(batch)
                loss = vortex_mae_paper_loss(pred_logits, gt_mask, alpha=2.0)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.detach()
            epoch_iou += calculate_iou(torch.sigmoid(pred_logits), gt_mask).detach()
            
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_iou, op=dist.ReduceOp.SUM)
        avg_loss = epoch_loss.item() / (len(train_loader) * world_size)
        avg_iou = epoch_iou.item() / (len(train_loader) * world_size)
        
        if rank == 0:
            print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | Mean IoU: {avg_iou:.4f}")
            if avg_iou > best_iou:
                best_iou = avg_iou
                torch.save({'model_state_dict': model.module.state_dict(), 'iou': best_iou}, 
                           os.path.join(args.save_dir, "vortexmae_finetuned_best.pth"))
        
        scheduler.step()
        
    if rank == 0: print("Fine-tuning Complete.")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
