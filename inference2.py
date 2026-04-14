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

def sliding_window_reconstruction(model, input_tensor, window_size=(128, 128, 128), overlap=0.75):
    """
    Sliding window for 3-channel velocity reconstruction with perfect grid alignment.
    This ensures derivatives (IVD) don't see any numerical 'seams' from patch boundaries.
    """
    device = input_tensor.device
    B, C, D, H, W = input_tensor.shape
    
    # 1. Stride calculation
    stride_d = max(1, int(window_size[0] * (1 - overlap)))
    stride_h = max(1, int(window_size[1] * (1 - overlap)))
    stride_w = max(1, int(window_size[2] * (1 - overlap)))
    
    # 2. Perfect Padding: 
    # Pad significantly to ensure the original volume is in the "steady-state" sum region.
    # We also pad to make the total volume a perfect multiple of stride + window_residue.
    pad_d = window_size[0]
    pad_h = window_size[1]
    pad_w = window_size[2]
    
    # Target padded dims
    def get_padded_dim(orig, pad, window, stride):
        target = orig + 2 * pad
        num_strides = (target - window + stride - 1) // stride
        return num_strides * stride + window
        
    D_p_target = get_padded_dim(D, pad_d, window_size[0], stride_d)
    H_p_target = get_padded_dim(H, pad_h, window_size[1], stride_h)
    W_p_target = get_padded_dim(W, pad_w, window_size[2], stride_w)
    
    # Actual padding added to the right to reach target
    p_d_extra = D_p_target - (D + 2 * pad_d)
    p_h_extra = H_p_target - (H + 2 * pad_h)
    p_w_extra = W_p_target - (W + 2 * pad_w)
    
    input_padded = F.pad(input_tensor, 
                         (pad_w, pad_w + p_w_extra, 
                          pad_h, pad_h + p_h_extra, 
                          pad_d, pad_d + p_d_extra), 
                         mode='replicate')
    
    _, _, D_p, H_p, W_p = input_padded.shape
    
    # 3. Pure Hann window, periodic=True perfectly sums to constant when overlapped correctly.
    window_d = torch.hann_window(window_size[0], periodic=True)
    window_h = torch.hann_window(window_size[1], periodic=True)
    window_w = torch.hann_window(window_size[2], periodic=True)
    blend_weight = window_d[:, None, None] * window_h[None, :, None] * window_w[None, None, :]
    blend_weight = blend_weight.unsqueeze(0).unsqueeze(0).to("cpu")
    
    out_recon_padded = torch.zeros((1, 3, D_p, H_p, W_p), dtype=torch.float32, device="cpu")
    overlap_count_padded = torch.zeros((1, 1, D_p, H_p, W_p), dtype=torch.float32, device="cpu")
    
    # 4. Iterate with fixed stride, NO min() logic (ensures math is perfect)
    for d_start in range(0, D_p - window_size[0] + 1, stride_d):
        for h_start in range(0, H_p - window_size[1] + 1, stride_h):
            for w_start in range(0, W_p - window_size[2] + 1, stride_w):
                
                d_end, h_end, w_end = d_start + window_size[0], h_start + window_size[1], w_start + window_size[2]
                crop = input_padded[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                
                # pad to multiple of 32 for Swin3D if window_size is not multiple (ours 128 is ok)
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        recon, _ = model(crop)
                
                recon = recon.cpu()
                out_recon_padded[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += recon * blend_weight
                overlap_count_padded[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += blend_weight

    # 5. Crop back to original physical volume. 
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
            recon_velocity = sliding_window_reconstruction(model, input_tensor, window_size=(128, 128, 128), overlap=0.75)
            
            # IVD Calculation
            # Original IVD (on normalized input)
            ivd_orig = calculate_ivd(input_tensor.cpu())[0] # [D, H, W]
            # Reconstructed IVD
            ivd_recon = calculate_ivd(recon_velocity)[0] # [D, H, W]
            
            # --- Noise Cleaning (Thresholding) ---
            val_max = ivd_recon.max()
            # Adaptive threshold: use a percentage of the max but don't let it kill everything
            # if the max is already small.
            eff_thresh = min(args.ivd_threshold, val_max * 0.1)
            ivd_recon[ivd_recon < eff_thresh] = 0.0
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
