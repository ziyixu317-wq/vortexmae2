import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pyvista as pv
from tqdm import tqdm

from dataset import VortexMAEDataset
from model import VortexMAE

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
            
            # 因为Swin Transformer具有层次化下采样结构，其输入尺寸必须是32（或者对应感受野）的倍数
            # 如果原始网格尺寸不是32的倍数，我们需要在右、下、后方进行零填充
            _, _, D, H, W = input_tensor.shape
            pad_d = (32 - D % 32) % 32
            pad_h = (32 - H % 32) % 32
            pad_w = (32 - W % 32) % 32
            
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                # F.pad format: (W_left, W_right, H_top, H_bottom, D_front, D_back)
                input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h, 0, pad_d))
                
            # 推理阶段
            with torch.cuda.amp.autocast():
                logits = model(input_tensor)
                probs = torch.sigmoid(logits)
            
            # 将输出裁剪回填充前的原始大小
            probs = probs[:, :, :D, :H, :W]
            
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
