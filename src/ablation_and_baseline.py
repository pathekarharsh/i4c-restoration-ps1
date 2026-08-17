import os
import csv
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.dataset import normalize, denormalize
from src.model import NAFNetLite
from src.metrices import MetricsCalculator

def main():
    # 1. Setup paths and device
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    val_csv = os.path.join(repo_root, "data", "val_pairs.csv")
    model_path = os.path.join(repo_root, "models", "model_best.pt")
    outputs_dir = os.path.join(repo_root, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using Device: {device}")

    # 2. Initialize Model and Metrics
    config_path = os.path.join(repo_root, "config.yaml")
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = NAFNetLite(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        num_features=config["model"]["num_features"],
        num_blocks=config["model"]["num_blocks"]
    ).to(device)
    
    print(f"Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metrics_calc = MetricsCalculator().to(device)

    # 3. Read Validation CSV
    pairs = []
    with open(val_csv, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pairs.append(row)

    print(f"Loaded {len(pairs)} validation pairs.")

    # 4. Metrics Storage
    bicubic_psnrs, bicubic_ssims, bicubic_lpipss = [], [], []
    model_psnrs, model_ssims, model_lpipss = [], [], []
    results = []

    # 5. Evaluation Loop
    with torch.no_grad():
        for pair in tqdm(pairs, desc="Evaluating Baseline vs Model"):
            gt_path = os.path.join(repo_root, pair["gt_path"])
            degraded_path = os.path.join(repo_root, pair["degraded_path"])
            
            # Load images
            gt_img = np.load(gt_path).astype(np.float32)
            degraded_img = np.load(degraded_path).astype(np.float32)
            
            # --- EVALUATE BICUBIC BASELINE ---
            # Upscale degraded LR (128x128) by 2x using standard Bicubic
            h, w = degraded_img.shape
            bicubic_img = cv2.resize(degraded_img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            
            # Normalize to [0, 1] range to evaluate metrics fairly
            bic_norm = normalize(bicubic_img)
            gt_norm = normalize(gt_img)
            
            # Convert to tensors
            bic_tensor = torch.tensor(bic_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            gt_tensor = torch.tensor(gt_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            # Compute baseline metrics
            bic_metrics = metrics_calc(bic_tensor, gt_tensor)
            bicubic_psnrs.append(bic_metrics["psnr"])
            bicubic_ssims.append(bic_metrics["ssim"])
            bicubic_lpipss.append(bic_metrics["lpips"])

            # --- EVALUATE MODEL ---
            # Normalize low-res input
            deg_norm = normalize(degraded_img)
            deg_tensor = torch.tensor(deg_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            # Forward pass
            model_out = model(deg_tensor)
            
            # Compute model metrics
            mod_metrics = metrics_calc(model_out, gt_tensor)
            model_psnrs.append(mod_metrics["psnr"])
            model_ssims.append(mod_metrics["ssim"])
            model_lpipss.append(mod_metrics["lpips"])
            
            # Record individual scores
            results.append({
                "gt_path": pair["gt_path"],
                "degraded_path": pair["degraded_path"],
                "bicubic_psnr": bic_metrics["psnr"],
                "bicubic_ssim": bic_metrics["ssim"],
                "model_psnr": mod_metrics["psnr"],
                "model_ssim": mod_metrics["ssim"],
                "model_lpips": mod_metrics["lpips"]
            })

    # 6. Compute Average Metrics
    avg_bic_psnr = np.mean(bicubic_psnrs)
    avg_bic_ssim = np.mean(bicubic_ssims)
    avg_bic_lpips = np.mean(bicubic_lpipss)

    avg_mod_psnr = np.mean(model_psnrs)
    avg_mod_ssim = np.mean(model_ssims)
    avg_mod_lpips = np.mean(model_lpipss)

    print("\n=============================================")
    print("        VAL SET BASELINE COMPARISON          ")
    print("=============================================")
    print(f"Bicubic Baseline:")
    print(f"  Avg PSNR:  {avg_bic_psnr:.4f} dB")
    print(f"  Avg SSIM:  {avg_bic_ssim:.4f}")
    print(f"  Avg LPIPS: {avg_bic_lpips:.4f}")
    print("---------------------------------------------")
    print(f"Our NAFNet-lite Model:")
    print(f"  Avg PSNR:  {avg_mod_psnr:.4f} dB")
    print(f"  Avg SSIM:  {avg_mod_ssim:.4f}")
    print(f"  Avg LPIPS: {avg_mod_lpips:.4f}")
    print("=============================================")

    # Write summary CSV
    summary_path = os.path.join(outputs_dir, "baseline_comparison.csv")
    with open(summary_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["method", "avg_psnr", "avg_ssim", "avg_lpips"])
        writer.writerow(["Bicubic Baseline", avg_bic_psnr, avg_bic_ssim, avg_bic_lpips])
        writer.writerow(["NAFNet-lite Model", avg_mod_psnr, avg_mod_ssim, avg_mod_lpips])
    print(f"Saved baseline metrics comparison to {summary_path}")

    # 7. Identify Failure Cases (Lowest Model PSNR/SSIM scores)
    # Sort results by model PSNR ascending (worst first)
    results_sorted = sorted(results, key=lambda x: x["model_psnr"])
    
    print("\nTop 3 Weakest validation outputs (Potential Failure Cases):")
    for idx in range(3):
        res = results_sorted[idx]
        print(f"  {idx+1}. File: {os.path.basename(res['gt_path'])} | Model PSNR: {res['model_psnr']:.4f} dB | Baseline PSNR: {res['bicubic_psnr']:.4f} dB")

    # Generate failure case visualization plots
    with torch.no_grad():
        for idx in range(2):
            res = results_sorted[idx]
            gt_path = os.path.join(repo_root, res["gt_path"])
            degraded_path = os.path.join(repo_root, res["degraded_path"])
            
            gt_img = np.load(gt_path).astype(np.float32)
            degraded_img = np.load(degraded_path).astype(np.float32)
            
            # Get model output
            deg_norm = normalize(degraded_img)
            deg_tensor = torch.tensor(deg_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            model_out = model(deg_tensor).squeeze(0).squeeze(0).cpu().numpy()
            
            # Denormalize to standard [0.0, 1.0] image range
            deg_display = denormalize(deg_norm)
            model_display = denormalize(model_out)
            gt_display = gt_img  # already in [0.0, 1.0] range
            
            # Save side-by-side visual comparison
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(deg_display, cmap="gray", vmin=0, vmax=1)
            axes[0].set_title(f"Input Degraded\n(Range: [{deg_display.min():.2f}, {deg_display.max():.2f}])")
            axes[0].axis("off")
            
            axes[1].imshow(model_display, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title(f"Model Restored (Failure Case)\nPSNR: {res['model_psnr']:.2f} dB | SSIM: {res['model_ssim']:.3f}")
            axes[1].axis("off")
            
            axes[2].imshow(gt_display, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title("Ground Truth target\n(Clean)")
            axes[2].axis("off")
            
            plt.tight_layout()
            out_img_path = os.path.join(outputs_dir, f"failure_case_{idx+1}.png")
            plt.savefig(out_img_path, dpi=150)
            plt.close()
            print(f"Saved failure case comparison plot {idx+1} to {out_img_path}")

if __name__ == "__main__":
    main()
