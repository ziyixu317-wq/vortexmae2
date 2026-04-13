
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from swin3d import SwinTransformer3D

class VortexMAE(nn.Module):
    """
    VortexMAE Foundation Model (Optimized for Small Data & T4 GPUs).
    Architecture: 3D Swin Transformer Encoder + 3D U-Net ConvTranspose Decoder.
    """
    def __init__(self, patch_size=(4, 4, 4), in_chans=3, out_chans=1,
                 embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], 
                 window_size=(4, 4, 4), mask_ratio=0.5, mode='pretrain'):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.mask_ratio = mask_ratio
        self.mode = mode
        
        # 1. Swin-ViT Encoder (Tiny+ Configuration)
        self.encoder = SwinTransformer3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, 
            depths=depths, num_heads=num_heads, window_size=window_size
        )
        
        # 2. Mask Token (MAE initialization)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 1, embed_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        # 3. Enhanced U-Net Decoder (Expansive Path with Transposed Convs)
        # Using concatenation for better feat preservation on small datasets
        d1, d2, d3, d4 = embed_dim, embed_dim*2, embed_dim*4, embed_dim*8
        
        # Stage 4 down -> 3: (d4 -> d3)
        self.up4 = nn.ConvTranspose3d(d4, d3, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv3d(d3 * 2, d3, kernel_size=3, padding=1),
            nn.BatchNorm3d(d3), nn.GELU()
        )
        
        # Stage 3 down -> 2: (d3 -> d2)
        self.up3 = nn.ConvTranspose3d(d3, d2, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv3d(d2 * 2, d2, kernel_size=3, padding=1),
            nn.BatchNorm3d(d2), nn.GELU()
        )
        
        # Stage 2 down -> 1: (d2 -> d1)
        self.up2 = nn.ConvTranspose3d(d2, d1, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv3d(d1 * 2, d1, kernel_size=3, padding=1),
            nn.BatchNorm3d(d1), nn.GELU()
        )
        
        # Stage 1 down -> Input Patch: (d1 -> in_chans)
        # patch_size=(4,4,4) requires 2x upsamples if we use stride=2, or one stride=4
        self.up_final = nn.Sequential(
            nn.ConvTranspose3d(d1, d1, kernel_size=2, stride=2),
            nn.Conv3d(d1, d1, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose3d(d1, d1, kernel_size=2, stride=2),
            nn.Conv3d(d1, d1, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        self.rec_head = nn.Conv3d(d1, in_chans, kernel_size=1)
        self.seg_head = nn.Conv3d(d1, out_chans, kernel_size=1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_embed = self.encoder.patch_embed(x)
        Bp, Dp, Hp, Wp, Cp = x_embed.shape
        N = Dp * Hp * Wp
        
        if self.mode == 'pretrain':
            noise = torch.rand(B, N, device=x.device)
            mask = (noise < self.mask_ratio).float().view(B, Dp, Hp, Wp, 1)
            x_masked = x_embed * (1 - mask) + self.mask_token * mask
            x_input = self.encoder.pos_drop(x_masked)
        elif self.mode == 'reconstruct':
            x_input = self.encoder.pos_drop(x_embed)
            mask = torch.ones(B, Dp, Hp, Wp, 1, device=x.device) # Keep mask as ones for compatibility
        else:
            x_input = self.encoder.pos_drop(x_embed)
            mask = None
        
        # Encoder stages
        outs = []
        curr_x = x_input
        for layer in self.encoder.layers:
            x_layer_out, curr_x = layer(curr_x)
            outs.append(x_layer_out.permute(0, 4, 1, 2, 3)) # (B, C, D, H, W)
            
        # Decoder stages with concatenation
        # outs[3]: d4, outs[2]: d3, outs[1]: d2, outs[0]: d1
        z = self.up4(outs[3])
        # Force same size for concat
        z = F.interpolate(z, size=outs[2].shape[2:], mode='trilinear', align_corners=False)
        z = torch.cat([z, outs[2]], dim=1)
        z = self.conv3(z)
        
        z = self.up3(z)
        z = F.interpolate(z, size=outs[1].shape[2:], mode='trilinear', align_corners=False)
        z = torch.cat([z, outs[1]], dim=1)
        z = self.conv2(z)
        
        z = self.up2(z)
        z = F.interpolate(z, size=outs[0].shape[2:], mode='trilinear', align_corners=False)
        z = torch.cat([z, outs[0]], dim=1)
        z = self.conv1(z)
        
        z = self.up_final(z)
        z = F.interpolate(z, size=(D, H, W), mode='trilinear', align_corners=False)
        
        if self.mode in ('pretrain', 'reconstruct'):
            out = self.rec_head(z)
            mask_px = F.interpolate(mask.permute(0, 4, 1, 2, 3), (D, H, W), mode='nearest')
            return out, mask_px
        else:
            return self.seg_head(z)

def vortex_mae_pretrain_loss(pred, target, mask):
    """MSE on masked regions."""
    loss = F.mse_loss(pred * mask, target * mask, reduction='sum')
    return loss / (mask.sum() * target.shape[1] + 1e-8)
