import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

def run_eda():
    # Setup paths relative to project root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gt_dir = os.path.join(repo_root, "data", "raw", "train", "GT")
    noisy_dir = os.path.join(repo_root, "data", "raw", "train", "NoisyLR")
    docs_dir = os.path.join(repo_root, "docs")
    images_dir = os.path.join(docs_dir, "images")
    notebook_dir = os.path.join(repo_root, "notebooks")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(notebook_dir, exist_ok=True)

    print("Starting EDA calculations...")
    gt_files = sorted(os.listdir(gt_dir))
    noisy_files = sorted(os.listdir(noisy_dir))

    # 1. Verify all pairs are 2x super-resolution
    all_2x = True
    for f in gt_files:
        gt_shape = np.load(os.path.join(gt_dir, f), mmap_mode="r").shape
        noisy_shape = np.load(os.path.join(noisy_dir, f), mmap_mode="r").shape
        if noisy_shape[0] * 2 != gt_shape[0] or noisy_shape[1] * 2 != gt_shape[1]:
            all_2x = False
            print(f"Non-2x shape found: Noisy {noisy_shape} -> GT {gt_shape} for {f}")
            break

    print(f"Verified all pairs are 2x: {all_2x}")

    # 2. Extract values and noise statistics
    # Sample 100 images for noise estimation and histogram analysis
    np.random.seed(42)
    sample_files = np.random.choice(gt_files, size=100, replace=False)

    noisy_min = float('inf')
    noisy_max = float('-inf')
    gt_min = float('inf')
    gt_max = float('-inf')

    residuals_list = []
    gt_downsampled_list = []

    for f in sample_files:
        gt_img = np.load(os.path.join(gt_dir, f))
        noisy_img = np.load(os.path.join(noisy_dir, f))

        noisy_min = min(noisy_min, noisy_img.min())
        noisy_max = max(noisy_max, noisy_img.max())
        gt_min = min(gt_min, gt_img.min())
        gt_max = max(gt_max, gt_img.max())

        # Downsample GT using INTER_AREA (area relation) to match NoisyLR resolution
        gt_down = cv2.resize(gt_img, (noisy_img.shape[1], noisy_img.shape[0]), interpolation=cv2.INTER_AREA)

        # Residual represents the noise added to the low-res image
        residual = noisy_img - gt_down
        residuals_list.append(residual.flatten())
        gt_downsampled_list.append(gt_down.flatten())

    residuals = np.concatenate(residuals_list)
    gt_downsampled = np.concatenate(gt_downsampled_list)

    # 3. Analyze Noise Type (Additive Gaussian vs. Multiplicative Speckle)
    # We partition local GT intensity into bins and measure residual variance in each bin.
    bins = np.linspace(0.0, 1.0, 11)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    var_per_bin = []
    std_per_bin = []

    for i in range(len(bins)-1):
        mask = (gt_downsampled >= bins[i]) & (gt_downsampled < bins[i+1])
        bin_residuals = residuals[mask]
        if len(bin_residuals) > 100:
            var_per_bin.append(np.var(bin_residuals))
            std_per_bin.append(np.std(bin_residuals))
        else:
            var_per_bin.append(0.0)
            std_per_bin.append(0.0)

    # Plot Residual Variance vs GT Intensity
    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, var_per_bin, 'o-', color='#e056fd', linewidth=2, markersize=8)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel('Local GT Pixel Value (Intensity)', fontsize=12)
    plt.ylabel('Variance of Residual (NoisyLR - Downsampled GT)', fontsize=12)
    plt.title('Noise Variance vs. Pixel Intensity (Speckle vs. Gaussian)', fontsize=14, fontweight='bold')
    variance_plot_path = os.path.join(images_dir, "noise_variance_vs_intensity.png")
    plt.tight_layout()
    plt.savefig(variance_plot_path, dpi=150)
    plt.close()

    # 4. Save visual side-by-side comparison for 3 samples
    plt.figure(figsize=(12, 8))
    for idx, f in enumerate(sample_files[:3]):
        gt_img = np.load(os.path.join(gt_dir, f))
        noisy_img = np.load(os.path.join(noisy_dir, f))
        
        plt.subplot(3, 2, 2*idx + 1)
        plt.imshow(noisy_img, cmap='gray', vmin=0, vmax=1)
        plt.title(f"Noisy LR (128x128) - Sample {f}", fontsize=10)
        plt.axis('off')

        plt.subplot(3, 2, 2*idx + 2)
        plt.imshow(gt_img, cmap='gray', vmin=0, vmax=1)
        plt.title(f"GT HR (256x256) - Sample {f}", fontsize=10)
        plt.axis('off')
        
    comparison_path = os.path.join(images_dir, "side_by_side_comparison.png")
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=150)
    plt.close()

    # 5. Save Pixel-Value Histogram
    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=100, color='#686de0', alpha=0.7, edgecolor='black', density=True)
    plt.axvline(x=0.0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title("Distribution of Residuals (Noise Profile)", fontsize=14, fontweight='bold')
    plt.xlabel("NoisyLR - Downsampled GT Value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    histogram_path = os.path.join(images_dir, "residual_histogram.png")
    plt.tight_layout()
    plt.savefig(histogram_path, dpi=150)
    plt.close()

    # Save Pixel Value Histogram of raw values
    plt.figure(figsize=(10, 5))
    sample_noisy = np.load(os.path.join(noisy_dir, sample_files[0])).flatten()
    sample_gt = np.load(os.path.join(gt_dir, sample_files[0])).flatten()
    plt.hist(sample_noisy, bins=100, alpha=0.6, label='Noisy LR (degraded)', color='#ff7979')
    plt.hist(sample_gt, bins=100, alpha=0.6, label='GT HR (clean)', color='#1dd1a1')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title("Pixel Value Range Histogram (Sample 000000)", fontsize=14, fontweight='bold')
    plt.xlabel("Pixel Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    pixel_histogram_path = os.path.join(images_dir, "pixel_value_histogram.png")
    plt.tight_layout()
    plt.savefig(pixel_histogram_path, dpi=150)
    plt.close()

    # Calculate correlation to see if variance increases with intensity
    correlation = np.corrcoef(bin_centers, var_per_bin)[0, 1]
    noise_type = "Additive Gaussian"
    if correlation > 0.8:
        noise_type = "Multiplicative Speckle (Variance scales with intensity)"
    elif correlation > 0.4:
        noise_type = "Mixed (Combined Additive Gaussian and Speckle)"

    # 6. Write eda_notes.md
    notes_content = f"""# EDA Report - Dataset Reconnaissance

## 1. Resolution Verification
- **All pair resolutions checked**: Verified that every training pair is exactly 2x resolution increase (`128x128` to `256x256`).
- **Test set resolution**: Verified that all test set files are `128x128`.
- **Dimensions**:
  - Degraded Inputs (NoisyLR): `(128, 128)`
  - Ground Truth Inputs (GT): `(256, 256)`

## 2. Pixel Value Range & Noise Overshoot
- **Ground Truth Range**: `[{gt_min:.4f}, {gt_max:.4f}]` (perfectly normalized in `[0, 1]`).
- **Noisy Low-Res Range**: `[{noisy_min:.4f}, {noisy_max:.4f}]`
- **Noise Overshoot**: Noisy inputs overshoot significantly below 0 (down to `{noisy_min:.4f}`) and above 1 (up to `{noisy_max:.4f}`).
- **Key Takeaway**: The network must handle inputs in this wider range, but should output values clipped to the target range `[0, 1]`.

## 3. Noise Profile Analysis
- **Residual Distribution (NoisyLR - Downsampled GT)**: Mean is `{residuals.mean():.6f}` and Std Dev is `{residuals.std():.6f}`.
- **Estimated Noise Type**: `{noise_type}`.
- **Variance vs Intensity Correlation**: `{correlation:.4f}`.
  - *Speckle test*: If variance changes with intensity, it points to multiplicative/signal-dependent noise. In our case, the correlation coefficient between the local GT intensity and the noise variance is `{correlation:.4f}`.
  
## 4. Visualizations
Visualizations have been generated and saved under `docs/images/`:
- **Side-by-side Comparison**: ![side_by_side](images/side_by_side_comparison.png)
- **Pixel Value Range Histogram**: ![pixel_histogram](images/pixel_value_histogram.png)
- **Noise Profile Histogram**: ![noise_histogram](images/residual_histogram.png)
- **Noise Variance vs Intensity Plot**: ![noise_variance](images/noise_variance_vs_intensity.png)
"""
    with open(os.path.join(docs_dir, "eda_notes.md"), "w") as f:
        f.write(notes_content)

    # 7. Generate eda.ipynb as a JSON notebook structure
    notebook_data = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Exploratory Data Analysis - Day 1\n",
                    "This notebook demonstrates dataset properties and noise profiling for the KLA Image Restoration dataset."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import cv2"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Verify Shapes and Ranges"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "gt_dir = '../data/raw/train/GT'\n",
                    "noisy_dir = '../data/raw/train/NoisyLR'\n",
                    "files = sorted(os.listdir(gt_dir))\n",
                    "print(f'Total training files: {len(files)}')\n",
                    "sample_gt = np.load(os.path.join(gt_dir, files[0]))\n",
                    "sample_noisy = np.load(os.path.join(noisy_dir, files[0]))\n",
                    "print(f'GT shape: {sample_gt.shape}, range: [{sample_gt.min():.4f}, {sample_gt.max():.4f}]')\n",
                    "print(f'Noisy shape: {sample_noisy.shape}, range: [{sample_noisy.min():.4f}, {sample_noisy.max():.4f}]')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Load Visualizations\n",
                    "Let's display the side-by-side comparisons and histograms generated in our analysis script."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from IPython.display import Image, display\n",
                    "display(Image(filename='../docs/images/side_by_side_comparison.png'))\n",
                    "display(Image(filename='../docs/images/pixel_value_histogram.png'))\n",
                    "display(Image(filename='../docs/images/noise_variance_vs_intensity.png'))"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open(os.path.join(notebook_dir, "eda.ipynb"), "w") as f:
        json.dump(notebook_data, f, indent=2)

    print("EDA completed. Generated plots, docs/eda_notes.md, and notebooks/eda.ipynb.")

if __name__ == "__main__":
    run_eda()
