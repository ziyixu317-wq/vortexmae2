import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pyvista as pv
from tqdm import tqdm

from dataset import VortexMAEDataset
from model import VortexMAE

def sliding_window_inference(model, input_tensor, window_size=(128, 128, 128), overlap=0.5):
    device = input_tensor.device
    B, C, D, H, W = input_tensor.shape
    
    stride_d = max(1, int(window_size[0] * (1 - overlap)))
    stride_h = max(1, int(window_size[1] * (1 - overlap)))
    stride_w = max(1, int(window_size[2] * (1 - overlap)))
    
    out_prob = torch.zeros((1, 1, D, H, W), dtype=torch.float32, device="cpu")
    overlap_count = torch.zeros((1, 1, D, H, W), dtype=torch.float32, device="cpu")
    
    for d in range(0, max(1, D - window_size[0] + stride_d), stride_d):
        for h in range(0, max(1, H - window_size[1] + stride_h), stride_h):
            for w in range(0, max(1, W - window_size[2] + stride_w), stride_w):
                d_start = min(d, max(0, D - window_size[0]))
                h_start = min(h, max(0, H - window_size[1]))
                w_start = min(w, max(0, W - window_size[2]))
                
                d_end = min(d_start + window_size[0], D)
                h_end = min(h_start + window_size[1], H)
                w_end = min(w_start + window_size[2], W)
                
                crop = input_tensor[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                
                _, _, cD, cH, cW = crop.shape
                pad_d_crop = (32 - cD % 32) % 32
                pad_h_crop = (32 - cH % 32) % 32
                pad_w_crop = (32 - cW % 32) % 32
                
                if pad_d_crop > 0 or pad_h_crop > 0 or pad_w_crop > 0:
                    crop = F.pad(crop, (0, pad_w_crop, 0, pad_h_crop, 0, pad_d_crop))
                
                with torch.amp.autocast('cuda'):
                    logits = model(crop)
                    probs = torch.sigmoid(logits)
                
                probs = probs[:, :, :cD, :cH, :cW].cpu()
                
                out_prob[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += probs
                overlap_count[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += 1.0

    return out_prob / overlap_count

def main():
    parser = argparse.ArgumentParser(description="VortexMAE Inference Script")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to your .vti data directory")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the finetuned model (.pth)")
    parser.add_argument("--save_dir", type=str, default="./results", help="Directory to save inference results")
    parser.add_argument("--threshold", type=float, default=0.5, help="Vortex detection probability threshold")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 初始化模型 (确保深度跟训练时修改过的代码同步 [2, 2, 12, 2])
    print("Building model...")
    model = VortexMAE(in_chans=3, out_chans=1, mode='segmentation', embed_dim=96, depths=[2, 2, 12, 2]).to(device)
    
    # 2. 加载微调权重
    print(f"Loading checkpoint from: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # 如果是用DDP训练出来的，字典里面的key可能会带"module."，这里去除它
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 3. 初始化数据集加载器
    # 对于 inference split, 数据集内部设置会自动不做随机裁剪 (do_crop=False)
    print(f"Scanning data from: {args.data_dir}")
    dataset = VortexMAEDataset(args.data_dir, split="inference", augment=False)
    
    print(f"Starting inference... found {len(dataset)} files.")
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            file_path = dataset.files[idx]
            filename = os.path.basename(file_path)
            
            # dataset 默认将流场处理为 [3, D, H, W] 的标准化张量
            input_tensor = dataset[idx].unsqueeze(0).to(device) # [1, 3, D, H, W]
            
            # 因为模型无法一次性塞下巨大的流场，使用滑动窗口进行切块推理
            # 这里的 window_size 设置为 (64, 128, 128) 兼顾大视野和 T4 16GB 显存的需求
            probs = sliding_window_inference(model, input_tensor, window_size=(64, 128, 128), overlap=0.25)

            
            # 根据阀值生成二值化的涡流掩码
            pred_mask = (probs > args.threshold).float().cpu().numpy()[0, 0] # 提取单批次单通道数据 [D, H, W]
            
            # ========================
            # 将识别结果写回到原 VTI 中保存
            # ========================
            mesh = pv.read(file_path)
            
            # 在 dataset.py 中的读取逻辑是： arr_3d = arr.reshape(dims[2], dims[1], dims[0])
            # 因此这里将其展平即可原路还原给 vtk 内部的 1D 数组索引顺序
            pred_mask_flat = pred_mask.flatten()
            probs_flat = probs.cpu().numpy()[0, 0].flatten()
            
            mesh.point_data["Pred_Vortex_Mask"] = pred_mask_flat
            mesh.point_data["Pred_Vortex_Prob"] = probs_flat
            
            out_path = os.path.join(args.save_dir, f"pred_{filename}")
            mesh.save(out_path)
            
    print(f"All done! Results saved in {args.save_dir}")

if __name__ == "__main__":
    main()
