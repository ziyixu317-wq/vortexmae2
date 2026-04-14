import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pyvista as pv
from tqdm import tqdm

from dataset import VortexMAEDataset
from model import VortexMAE
from vortex_utils import calculate_ivd

def sliding_window_reconstruction(model, input_tensor, window_size=(128, 128, 128), overlap=0.5):
    """
    Sliding window for 3-channel velocity reconstruction with physical edge padding.
    """
    device = input_tensor.device
    B, C, D, H, W = input_tensor.shape
    
    stride_d = max(1, int(window_size[0] * (1 - overlap)))
    stride_h = max(1, int(window_size[1] * (1 - overlap)))
    stride_w = max(1, int(window_size[2] * (1 - overlap)))
    
    # 1. Pad the actual physical volume so that the original volume 
    # sits exactly in the middle of overlapping pure Hann windows, 
    # making the overlap_count strictly uniform (1.0).
    pad_d = window_size[0] // 2
    pad_h = window_size[1] // 2
    pad_w = window_size[2] // 2
    
    input_padded = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d), mode='replicate')
    B_p, C_p, D_p, H_p, W_p = input_padded.shape
    
    # 2. Pure Hann window, periodic=True perfectly sums to 1.0 at 50% overlap
    window_d = torch.hann_window(window_size[0], periodic=True)
    window_h = torch.hann_window(window_size[1], periodic=True)
    window_w = torch.hann_window(window_size[2], periodic=True)
    blend_weight = window_d[:, None, None] * window_h[None, :, None] * window_w[None, None, :]
    blend_weight = blend_weight.unsqueeze(0).unsqueeze(0).to("cpu")
    
    out_recon_padded = torch.zeros((1, 3, D_p, H_p, W_p), dtype=torch.float32, device="cpu")
    overlap_count_padded = torch.zeros((1, 1, D_p, H_p, W_p), dtype=torch.float32, device="cpu")
    
    for d in range(0, max(1, D_p - window_size[0] + stride_d), stride_d):
        for h in range(0, max(1, H_p - window_size[1] + stride_h), stride_h):
            for w in range(0, max(1, W_p - window_size[2] + stride_w), stride_w):
                d_start = min(d, max(0, D_p - window_size[0]))
                h_start = min(h, max(0, H_p - window_size[1]))
                w_start = min(w, max(0, W_p - window_size[2]))
                
                d_end = min(d_start + window_size[0], D_p)
                h_end = min(h_start + window_size[1], H_p)
                w_end = min(w_start + window_size[2], W_p)
                
                crop = input_padded[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                
                _, _, cD, cH, cW = crop.shape
                # pad to multiple of 32 for Swin3D
                pad_d_crop = (32 - cD % 32) % 32
                pad_h_crop = (32 - cH % 32) % 32
                pad_w_crop = (32 - cW % 32) % 32
                
                if pad_d_crop > 0 or pad_h_crop > 0 or pad_w_crop > 0:
                    crop = F.pad(crop, (0, pad_w_crop, 0, pad_h_crop, 0, pad_d_crop))
                
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        recon, _ = model(crop)
                
                recon = recon[:, :, :cD, :cH, :cW].cpu()
                current_weight = blend_weight[:, :, :cD, :cH, :cW]
                
                out_recon_padded[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += recon * current_weight
                overlap_count_padded[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += current_weight

    # 3. Crop back to original physical volume. 
    # Notice how we avoid the un-overlapped tapering edges completely!
    out_recon = out_recon_padded[:, :, pad_d:pad_d+D, pad_h:pad_h+H, pad_w:pad_w+W]
    overlap_count = overlap_count_padded[:, :, pad_d:pad_d+D, pad_h:pad_h+H, pad_w:pad_w+W]
    
    overlap_count = torch.clamp(overlap_count, min=1e-8)
    return out_recon / overlap_count
def main():
    parser = argparse.ArgumentParser(description="VortexMAE Reconstruction & IVD Script")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to .vti data")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to pre-trained model (.pth)")
    parser.add_argument("--save_dir", type=str, default="./results_recon", help="Save directory")
    parser.add_argument("--mask_ratio", type=float, default=0.0, help="Ratio of tokens to mask during reconstruction")
    parser.add_argument("--max_files", type=int, default=None, help="Maximum number of files to process")
    parser.add_argument("--select_files", type=str, default=None, help="Specific filenames")
    parser.add_argument("--ivd_threshold", type=float, default=0.04, help="Soft threshold for IVD cleaning (e.g. 0.04-0.1)")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Model in Reconstruct Mode
    print("Building model (Reconstruct Mode)...")
    model = VortexMAE(in_chans=3, out_chans=1, mode='reconstruct', 
                      embed_dim=96, depths=[2, 2, 12, 2], 
                      mask_ratio=0.0).to(device)
    
    # 2. Load Weights
    print(f"Loading checkpoint from: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 3. Initialize Dataset (Using split='all' for full directory processing)
    print(f"Scanning data from: {args.data_dir}")
    dataset = VortexMAEDataset(args.data_dir, split="all", augment=False)
    dataset.do_crop = False # Ensure full volume is returned for inference

    # --- Selective Processing Logic ---
    if args.select_files:
        selected = [s.strip() for s in args.select_files.split(",")]
        dataset.files = [f for f in dataset.files if os.path.basename(f) in selected]
    
    if args.max_files is not None:
        dataset.files = dataset.files[:args.max_files]
    # ---------------------------------
    
    print(f"Starting reconstruction... found {len(dataset)} files after filtering.")
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            file_path = dataset.files[idx]
            filename = os.path.basename(file_path)
            
            # dataset 默认将流场处理为 [3, D, H, W] 的标准化张量
            input_tensor = dataset[idx].unsqueeze(0).to(device) # [1, 3, D, H, W]
            
            # Reconstruction via sliding window
            # Use 128x128x128 to match training crop size, with 50% overlap
            recon_velocity = sliding_window_reconstruction(model, input_tensor, window_size=(128, 128, 128), overlap=0.5)
            
            # IVD Calculation
            # Original IVD (on normalized input)
            ivd_orig = calculate_ivd(input_tensor.cpu())[0] # [D, H, W]
            # Reconstructed IVD
            ivd_recon = calculate_ivd(recon_velocity)[0] # [D, H, W]
            
            # --- Noise Cleaning (Thresholding) ---
            # Set values below threshold to 0 to keep background clean
            # We scale the threshold by the local max to be adaptive but keep a floor
            val_max = ivd_recon.max()
            effective_thresh = max(args.ivd_threshold, val_max * 0.05)
            ivd_recon[ivd_recon < effective_thresh] = 0.0
            # -------------------------------------
            
            # Save results to VTI
            mesh = pv.read(file_path)
            
            # Normalized Reconstruction Components
            u_rec = recon_velocity[0, 0].numpy().flatten()
            v_rec = recon_velocity[0, 1].numpy().flatten()
            w_rec = recon_velocity[0, 2].numpy().flatten()
            
            mesh.point_data["u_rec"] = u_rec
            mesh.point_data["v_rec"] = v_rec
            mesh.point_data["w_rec"] = w_rec
            mesh.point_data["IVD_Original_Norm"] = ivd_orig.numpy().flatten()
            mesh.point_data["IVD_Reconstructed"] = ivd_recon.numpy().flatten()
            
            out_path = os.path.join(args.save_dir, f"recon_{filename}")
            mesh.save(out_path)
            
    print(f"All done! Results saved in {args.save_dir}")

if __name__ == "__main__":
    main()
