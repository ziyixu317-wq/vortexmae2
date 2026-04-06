
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath, trunc_normal_

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x); x = self.fc2(x); x = self.drop(x)
        return x

def window_partition3d(x, window_size):
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], C)
    return windows

def window_reverse3d(windows, window_size, B, D, H, W):
    windows = windows.view(-1, window_size[0], window_size[1], window_size[2], windows.shape[-1])
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))
        coords_d = torch.arange(self.window_size[0]); coords_h = torch.arange(self.window_size[1]); coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1; relative_coords[:, :, 1] += self.window_size[1] - 1; relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1); relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias); self.attn_drop = nn.Dropout(attn_drop); self.proj = nn.Linear(dim, dim); self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale; attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        attn = attn + relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]; attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0); attn = attn.view(-1, self.num_heads, N, N)
        attn = torch.softmax(attn, dim=-1); attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C); x = self.proj(x); x = self.proj_drop(x)
        return x

class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=(4, 4, 4), shift_size=(0, 0, 0), mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim; self.num_heads = num_heads; self.window_size = window_size; self.shift_size = shift_size
        self.norm1 = norm_layer(dim); self.attn = WindowAttention3D(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity(); self.norm2 = norm_layer(dim); self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
    def forward(self, x):
        B, D, H, W, C = x.shape
        shortcut = x; x = self.norm1(x)
        pad_d1 = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_r = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        if any(i > 0 for i in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            img_mask = torch.zeros((1, Dp, Hp, Wp, 1), device=x.device)
            d_slices = (slice(0, -self.window_size[0]), slice(-self.window_size[0], -self.shift_size[0]), slice(-self.shift_size[0], None))
            h_slices = (slice(0, -self.window_size[1]), slice(-self.window_size[1], -self.shift_size[1]), slice(-self.shift_size[1], None))
            w_slices = (slice(0, -self.window_size[2]), slice(-self.window_size[2], -self.shift_size[2]), slice(-self.shift_size[2], None))
            cnt = 0
            for d in d_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, d, h, w, :] = cnt; cnt += 1
            mask_windows = window_partition3d(img_mask, self.window_size).view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            shifted_x = x; attn_mask = None
        attn_windows = self.attn(window_partition3d(shifted_x, self.window_size), mask=attn_mask)
        shifted_x = window_reverse3d(attn_windows, self.window_size, B, Dp, Hp, Wp)
        if any(i > 0 for i in self.shift_size): x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else: x = shifted_x
        if pad_d1 > 0 or pad_b > 0 or pad_r > 0: x = x[:, :D, :H, :W, :].contiguous()
        x = shortcut + self.drop_path(x); x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMerging3D(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__(); self.dim = dim; self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False); self.norm = norm_layer(8 * dim)
    def forward(self, x):
        B, D, H, W, C = x.shape
        if (H % 2 == 1) or (W % 2 == 1) or (D % 2 == 1): x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, D % 2))
        x = torch.cat([x[:, 0::2, 0::2, 0::2, :], x[:, 1::2, 0::2, 0::2, :], x[:, 0::2, 1::2, 0::2, :], x[:, 0::2, 0::2, 1::2, :], x[:, 1::2, 1::2, 0::2, :], x[:, 1::2, 0::2, 1::2, :], x[:, 0::2, 1::2, 1::2, :], x[:, 1::2, 1::2, 1::2, :]], -1)
        return self.reduction(self.norm(x))

class BasicLayer3D(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([SwinTransformerBlock3D(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2), mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample is not None else None
    def forward(self, x):
        for blk in self.blocks: x = blk(x)
        if self.downsample is not None: return x, self.downsample(x)
        return x, x

class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(4,4,4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__(); self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size); self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x): x = self.proj(x); return self.norm(rearrange(x, 'b c d h w -> b d h w c'))

class SwinTransformer3D(nn.Module):
    def __init__(self, patch_size=(4,4,4), in_chans=3, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=(4, 4, 4), mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=nn.LayerNorm, patch_norm=True):
        super().__init__(); self.num_layers = len(depths); self.embed_dim = embed_dim; self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patch_embed = PatchEmbed3D(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None); self.pos_drop = nn.Dropout(p=drop_rate); dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList([BasicLayer3D(dim=int(embed_dim * 2 ** i), depth=depths[i], num_heads=num_heads[i], window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])], norm_layer=norm_layer, downsample=PatchMerging3D if (i < self.num_layers - 1) else None) for i in range(self.num_layers)])
        self.norm = norm_layer(self.num_features)
    def forward(self, x):
        x = self.pos_drop(self.patch_embed(x)); outs = []
        for layer in self.layers: x_out, x = layer(x); outs.append(x_out)
        return self.norm(x), outs
