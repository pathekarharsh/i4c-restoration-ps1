import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src.dataset import KLANDataset, denormalize

def run_sanity_check():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_csv = os.path.join(repo_root, "data", "train_pairs.csv")
    val_csv = os.path.join(repo_root, "data", "val_pairs.csv")
    output_img = os.path.join(repo_root, "docs", "images", "sanity_check.png")

    print(f"Loading datasets...")
    train_dataset = KLANDataset(train_csv, is_train=True, synthetic_prob=0.5)
    val_dataset = KLANDataset(val_csv, is_train=False)

    print(f"Dataset sizes:")
    print(f"  Train: {len(train_dataset)} pairs")
    print(f"  Val: {len(val_dataset)} pairs")

    # Verify a single item
    degraded, gt = train_dataset[0]
    print("\nSample 0 dimensions:")
    print(f"  Degraded: shape={degraded.shape}, dtype={degraded.dtype}, range=[{degraded.min():.4f}, {degraded.max():.4f}]")
    print(f"  GT:       shape={gt.shape}, dtype={gt.dtype}, range=[{gt.min():.4f}, {gt.max():.4f}]")

    # Set up DataLoader
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    batch_degraded, batch_gt = next(iter(loader))
    
    print("\nDataLoader batch dimensions:")
    print(f"  Batch Degraded: shape={batch_degraded.shape}")
    print(f"  Batch GT:       shape={batch_gt.shape}")

    # Plot 4 samples side-by-side
    plt.figure(figsize=(10, 10))
    for i in range(4):
        # Denormalize to show raw physical pixel values in plot
        deg_img = denormalize(batch_degraded[i, 0].numpy())
        gt_img = denormalize(batch_gt[i, 0].numpy())
        
        # Degraded LR input (128x128)
        plt.subplot(4, 2, 2*i + 1)
        plt.imshow(deg_img, cmap="gray", vmin=0, vmax=1)
        plt.title(f"Input Degraded (128x128) - Min: {deg_img.min():.2f}, Max: {deg_img.max():.2f}", fontsize=8)
        plt.axis("off")
        
        # Ground Truth HR target (256x256)
        plt.subplot(4, 2, 2*i + 2)
        plt.imshow(gt_img, cmap="gray", vmin=0, vmax=1)
        plt.title(f"GT Clean HR (256x256) - Min: {gt_img.min():.2f}, Max: {gt_img.max():.2f}", fontsize=8)
        plt.axis("off")
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.savefig(output_img, dpi=150)
    plt.close()
    
    print(f"\nVisual sanity check grid saved to: {output_img}")
    print("Sanity check completed successfully.")

if __name__ == "__main__":
    run_sanity_check()
