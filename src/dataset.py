import os
import csv
import random
import numpy as np
import torch
from torch.utils.data import Dataset
# (Synthetic degradation imported lazily below to optimize evaluation script startup time)

# Robust normalization parameters based on 1st and 99th percentiles of the training dataset.
# Using min_val = -0.15 and max_val = 1.15 to preserve noise overshoot symmetrically and prevent bias.
MIN_VAL = -0.15
MAX_VAL = 1.15

def normalize(img, min_val=MIN_VAL, max_val=MAX_VAL):
    """
    Normalizes pixel values from the [min_val, max_val] range to [0.0, 1.0].
    """
    clipped = np.clip(img, min_val, max_val)
    return (clipped - min_val) / (max_val - min_val)

def denormalize(img, min_val=MIN_VAL, max_val=MAX_VAL):
    """
    Denormalizes pixel values from [0.0, 1.0] back to the original range.
    Works for both NumPy arrays and PyTorch tensors.
    """
    return img * (max_val - min_val) + min_val

class KLANDataset(Dataset):
    def __init__(self, csv_file, is_train=True, synthetic_prob=0.5):
        """
        csv_file: path to train_pairs.csv or val_pairs.csv
        is_train: if True, applies random spatial augmentations and online synthetic degradation
        synthetic_prob: probability of generating synthetic degradation instead of loading real pair
        """
        self.is_train = is_train
        self.synthetic_prob = synthetic_prob
        self.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.cache_in_memory = True
        self.gt_cache = {}
        self.degraded_cache = {}
        
        self.pairs = []
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Index CSV not found at {csv_file}")
            
        with open(csv_file, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.pairs.append(row)
                
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Resolve absolute paths
        gt_path = os.path.join(self.repo_root, pair["gt_path"])
        degraded_path = os.path.join(self.repo_root, pair["degraded_path"])
        
        # Load clean HR GT image
        if self.cache_in_memory and idx in self.gt_cache:
            gt_img = self.gt_cache[idx]
        else:
            gt_img = np.load(gt_path).astype(np.float32)
            if self.cache_in_memory:
                self.gt_cache[idx] = gt_img
        
        # Decide if we should generate synthetic degradation or use real pair (only during training)
        use_synthetic = self.is_train and (random.random() < self.synthetic_prob)
        
        if use_synthetic:
            # Generate synthetic degraded image online from the clean GT HR image
            from src.synthetic_degrade import degrade_image
            degraded_img = degrade_image(gt_img)
        else:
            # Load real degraded LR image
            if self.cache_in_memory and idx in self.degraded_cache:
                degraded_img = self.degraded_cache[idx]
            else:
                degraded_img = np.load(degraded_path).astype(np.float32)
                if self.cache_in_memory:
                    self.degraded_cache[idx] = degraded_img
            
        # Apply Spatial Augmentations (Only during training)
        if self.is_train:
            # Random Horizontal Flip
            if random.random() < 0.5:
                gt_img = np.fliplr(gt_img).copy()
                degraded_img = np.fliplr(degraded_img).copy()
                
            # Random Vertical Flip
            if random.random() < 0.5:
                gt_img = np.flipud(gt_img).copy()
                degraded_img = np.flipud(degraded_img).copy()
                
            # Random 90-degree rotations (0, 90, 180, 270 degrees)
            rot_k = random.choice([0, 1, 2, 3])
            if rot_k > 0:
                gt_img = np.rot90(gt_img, rot_k).copy()
                degraded_img = np.rot90(degraded_img, rot_k).copy()
                
        # Normalize both inputs and ground-truth values to [0.0, 1.0]
        gt_norm = normalize(gt_img)
        degraded_norm = normalize(degraded_img)
        
        # Convert to PyTorch float32 tensors with channel dimension (1, H, W)
        gt_tensor = torch.tensor(gt_norm, dtype=torch.float32).unsqueeze(0)
        degraded_tensor = torch.tensor(degraded_norm, dtype=torch.float32).unsqueeze(0)
        
        return degraded_tensor, gt_tensor
