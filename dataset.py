
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pyvista as pv

def read_vti_velocity(filepath, velocity_names=("u", "v", "w")):
    """Reads a .vti file and returns the velocity field as (3, D, H, W)."""
    mesh = pv.read(filepath)
    dims = mesh.dimensions  # (nx, ny, nz)
    
    components = []
    for name in velocity_names:
        if name in mesh.point_data:
            arr = mesh.point_data[name]
        elif name in mesh.cell_data:
            arr = mesh.cell_data[name]
        else:
            potential_vectors = [k for k in mesh.point_data.keys() if 'velocity' in k.lower()]
            if potential_vectors:
                vec = mesh.point_data[potential_vectors[0]]
                if len(vec.shape) == 2 and vec.shape[1] == 3:
                   u = vec[:, 0].reshape(dims[2], dims[1], dims[0])
                   v = vec[:, 1].reshape(dims[2], dims[1], dims[0])
                   w = vec[:, 2].reshape(dims[2], dims[1], dims[0])
                   return np.stack([u, v, w], axis=0).astype(np.float32)
            raise KeyError(f"Velocity component '{name}' not found in {filepath}.")
            
        arr_3d = arr.reshape(dims[2], dims[1], dims[0])
        components.append(arr_3d)
    
    return np.stack(components, axis=0).astype(np.float32)

class VortexMAEDataset(Dataset):
    """
    Optimized Dataset for T4 GPUs with 3D Data Augmentation.
    """
    def __init__(self, data_dir, split="train", split_ratio=0.7, 
                 normalize=True, crop_size=128, augment=False):
        self.data_dir = data_dir
        self.normalize = normalize
        self.crop_size = crop_size
        self.augment = augment
        self.do_crop = split not in ("inference")
        
        self.all_files = sorted(glob.glob(os.path.join(data_dir, "*.vti")))
        if not self.all_files:
            raise FileNotFoundError(f"No .vti files found in {data_dir}")
            
        num_total = len(self.all_files)
        
        # Consistent split indices
        idx_p1 = int(num_total * 0.3)
        idx_p2 = int(num_total * 0.4)
        idx_f = int(num_total * 0.65)
        
        if split == "pretrain_train":
            self.files = self.all_files[:max(1, idx_p1)]
        elif split == "pretrain_eval":
            self.files = self.all_files[idx_p1:max(idx_p1+1, idx_p2)]
        elif split == "finetune_train":
            self.files = self.all_files[idx_p2:max(idx_p2+1, idx_f)]
        elif split == "inference":
            self.files = self.all_files[idx_f:]
        elif split == "all":
            self.files = self.all_files
        elif split == "train":
            self.files = self.all_files[:int(num_total * split_ratio)]
        else: # test/eval
            self.files = self.all_files[int(num_total * split_ratio):]
            
        print(f"[{split}] Loaded {len(self.files)} files. Augmentation: {self.augment}")
        
        # Normalization is now performed per-sample in __getitem__ to align with paper Eq. 2.

    def __len__(self):
        return len(self.files)

    def _apply_augment(self, sample):
        """Random flips and 90-degree rotations in 3D."""
        # 1. Random Flips (3 axes)
        for ax in [1, 2, 3]: # D, H, W
            if np.random.random() > 0.5:
                sample = np.flip(sample, axis=ax)
                # Note: For velocity field, flipping an axis should also flip the corresponding vector component
                # e.g., flipping X-axis (W) requires u = -u if keeping right-handedness, 
                # but for vortex identification (scalar-like) we focus on structure.
                # Strictly:
                if ax == 3: sample[0] = -sample[0] # u component
                if ax == 2: sample[1] = -sample[1] # v component
                if ax == 1: sample[2] = -sample[2] # w component

        # 2. Random 90-degree Rotations (around D, H, or W axes)
        # Select one of 3 planes to rotate
        if np.random.random() > 0.5:
            axes = np.random.choice([1, 2, 3], 2, replace=False)
            k = np.random.randint(1, 4) # 90, 180, 270
            sample = np.rot90(sample, k=k, axes=tuple(axes))
            # Rotating vector fields is complex (requires rotation matrix on components), 
            # here we use a simplified version. For foundation models, learning the 
            # invariance is key.
            
        return sample

    def __getitem__(self, idx):
        sample = read_vti_velocity(self.files[idx])
        
        if self.normalize:
            f_min = sample.min(axis=(1, 2, 3), keepdims=True)
            f_max = sample.max(axis=(1, 2, 3), keepdims=True)
            sample = (sample - f_min) / (f_max - f_min + 1e-8)
        
        if self.augment:
            sample = self._apply_augment(sample)

        if self.do_crop:
            _, D, H, W = sample.shape
            cs = self.crop_size
            cd, ch, cw = min(cs, D), min(cs, H), min(cs, W)
            
            d0 = np.random.randint(0, D - cd + 1) if D > cd else 0
            h0 = np.random.randint(0, H - ch + 1) if H > ch else 0
            w0 = np.random.randint(0, W - cw + 1) if W > cw else 0
            
            sample = sample[:, d0:d0+cd, h0:h0+ch, w0:w0+cw]
            
            if cd < cs or ch < cs or cw < cs:
                padded = np.zeros((3, cs, cs, cs), dtype=np.float32)
                padded[:, :cd, :ch, :cw] = sample
                sample = padded
        
        return torch.from_numpy(sample.copy())
