import os
import csv
import yaml
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.dataset import KLANDataset, denormalize
from src.model import NAFNetLite
from src.losses import HybridLoss
from src.metrices import MetricsCalculator

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EarlyStopping:
    """
    Early stopping helper that monitors validation SSIM and stops training if it doesn't improve.
    """
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"Early Stopping: No improvement for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0

def save_progress_visualization(model, val_dataset, epoch, device, output_path):
    """
    Generates and saves a side-by-side comparison (Noisy Input, Prediction, GT) 
    for 4 fixed validation images to visually monitor restoration progression.
    """
    model.eval()
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    
    # Use deterministic indices for progress tracking (first 4 validation images)
    with torch.no_grad():
        for i in range(4):
            degraded, gt = val_dataset[i]
            
            # Forward pass through model
            inputs = degraded.unsqueeze(0).to(device)  # add batch dimension
            pred = model(inputs).squeeze(0).cpu()      # remove batch dimension
            
            # Denormalize to show raw physical pixel values
            deg_img = denormalize(degraded[0].numpy())
            pred_img = denormalize(pred[0].numpy())
            gt_img = denormalize(gt[0].numpy())
            
            # Plot 1: Input Degraded
            axes[i, 0].imshow(deg_img, cmap="gray", vmin=0, vmax=1)
            axes[i, 0].set_title(f"Input Degraded (128x128)", fontsize=10)
            axes[i, 0].axis("off")
            
            # Plot 2: Model Restored Prediction
            axes[i, 1].imshow(pred_img, cmap="gray", vmin=0, vmax=1)
            axes[i, 1].set_title(f"Model Prediction (256x256)", fontsize=10)
            axes[i, 1].axis("off")
            
            # Plot 3: Ground Truth Target
            axes[i, 2].imshow(gt_img, cmap="gray", vmin=0, vmax=1)
            axes[i, 2].set_title(f"GT Clean HR (256x256)", fontsize=10)
            axes[i, 2].axis("off")
            
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved validation progress visualization for Epoch {epoch} to {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train KLA Image Restoration Model")
    parser.add_argument("--dry_run", action="store_true", help="Run 2 steps per epoch for verification")
    args, unknown = parser.parse_known_args()

    # 1. Setup paths and config
    repo_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(repo_root, "config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set reproducibility seed
    set_seed(config.get("seed", 42))

    # Set up directories
    weights_dir = os.path.join(repo_root, "models")
    outputs_dir = os.path.join(repo_root, "outputs")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    # CSV log path
    log_csv_path = os.path.join(outputs_dir, "training_log.csv")
    write_header = not os.path.exists(log_csv_path)

    # 2. Configure device (CUDA, Apple MPS, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using Device: CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using Device: CPU")

    # 3. Datasets and DataLoaders
    train_csv = os.path.join(repo_root, config["dataset"]["train_pairs_csv"])
    val_csv = os.path.join(repo_root, config["dataset"]["val_pairs_csv"])
    
    batch_size = config["training"]["batch_size"]
    print(f"Loading datasets...")
    train_dataset = KLANDataset(train_csv, is_train=True, synthetic_prob=0.5)
    val_dataset = KLANDataset(val_csv, is_train=False)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )

    print(f"  Training pairs:   {len(train_dataset)}")
    print(f"  Validation pairs: {len(val_dataset)}")

    # 4. Initialize model, optimizer, scheduler, loss, metrics, early stopping
    model = NAFNetLite(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        num_features=config["model"]["num_features"],
        num_blocks=config["model"]["num_blocks"]
    ).to(device)
    
    optimizer = AdamW(
        model.parameters(), 
        lr=config["training"]["learning_rate"], 
        weight_decay=1e-4
    )
    
    epochs = config["training"]["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    criterion = HybridLoss().to(device)
    metrics_calc = MetricsCalculator().to(device)

    # Initialize early stopping on validation SSIM
    patience = config["training"].get("early_stopping_patience", 15)
    early_stop = EarlyStopping(patience=patience)

    # 5. Mixed Precision Setup (only enabled on CUDA for stability)
    use_amp = config["training"].get("mixed_precision", True) and device.type == "cuda"
    scaler = None
    if use_amp:
        try:
            scaler = torch.amp.GradScaler()
            print("AMP Mixed Precision enabled (CUDA, torch.amp.GradScaler).")
        except AttributeError:
            try:
                scaler = torch.cuda.amp.GradScaler()
                print("AMP Mixed Precision enabled (CUDA, torch.cuda.amp.GradScaler).")
            except Exception as e:
                use_amp = False
                print(f"AMP disabled due to initialization error: {e}")

    # Best metric tracker
    best_psnr = -1.0

    # 6. Main Training Loop
    print(f"\nStarting training loop for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for idx, (inputs, targets) in enumerate(loop):
            if args.dry_run and idx >= 2:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
                assert scaler is not None
                try:
                    autocast_ctx = torch.amp.autocast(device_type="cuda")
                except AttributeError:
                    autocast_ctx = torch.cuda.amp.autocast()
                
                with autocast_ctx:
                    outputs = model(inputs)
                    loss, logs = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss, logs = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        scheduler.step()
        num_steps = 2 if args.dry_run else len(train_loader)
        avg_train_loss = train_loss / num_steps
        
        # 7. Validation Loop
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        val_lpips = 0.0
        
        print(f"Running validation for Epoch {epoch}...")
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(val_loader):
                if args.dry_run and idx >= 2:
                    break
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss, logs = criterion(outputs, targets)
                val_loss += loss.item()
                
                # Calculate metrics
                metrics = metrics_calc(outputs, targets)
                val_psnr += metrics["psnr"]
                val_ssim += metrics["ssim"]
                val_lpips += metrics["lpips"]

        num_val_steps = 2 if args.dry_run else len(val_loader)
        avg_val_loss = val_loss / num_val_steps
        avg_val_psnr = val_psnr / num_val_steps
        avg_val_ssim = val_ssim / num_val_steps
        avg_val_lpips = val_lpips / num_val_steps

        print(f"\n--- Epoch {epoch} Summary ---")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {avg_val_loss:.6f}")
        print(f"  Val PSNR:   {avg_val_psnr:.4f} dB")
        print(f"  Val SSIM:   {avg_val_ssim:.4f}")
        print(f"  Val LPIPS:  {avg_val_lpips:.4f}")
        print("--------------------------")

        # 8. Logging to CSV
        with open(log_csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(["epoch", "train_loss", "val_loss", "val_psnr", "val_ssim", "val_lpips"])
                write_header = False
            writer.writerow([epoch, avg_train_loss, avg_val_loss, avg_val_psnr, avg_val_ssim, avg_val_lpips])
        print(f"Logged epoch metrics to {log_csv_path}")

        # 9. Save Visual Progress Comparison
        progress_img_path = os.path.join(outputs_dir, f"val_progress_epoch_{epoch}.png")
        save_progress_visualization(model, val_dataset, epoch, device, progress_img_path)

        # 10. Checkpointing
        checkpoint_path = os.path.join(weights_dir, f"model_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_psnr': avg_val_psnr,
            'val_ssim': avg_val_ssim,
            'val_lpips': avg_val_lpips
        }, checkpoint_path)

        # Save best checkpoint
        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            best_checkpoint_path = os.path.join(weights_dir, "model_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': avg_val_psnr,
                'val_ssim': avg_val_ssim,
                'val_lpips': avg_val_lpips
            }, best_checkpoint_path)
            print(f"  *** New best model saved with PSNR: {best_psnr:.4f} dB ***")

        # 11. Early Stopping Check
        early_stop(avg_val_ssim)
        if early_stop.should_stop:
            print(f"Early stopping triggered. Training stopped at epoch {epoch}.")
            break

    print("\nTraining completed.")

if __name__ == "__main__":
    main()
