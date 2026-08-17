# KLA Image Restoration Hackathon — NAFNet-lite Pipeline

This repository implements a lightweight, high-performance image restoration (denoising + 2x super-resolution) pipeline built for the **KLA Image Restoration Challenge**. 

The solution uses a **Nonlinear Activation-Free Network (NAFNet-lite)** architecture (~2.3M parameters) optimized to perform joint upscaling and signal-dependent speckle denoising. It features robust percentile-based normalization to address high noise overshoots and utilizes a hybrid loss function combining pixel-fidelity, structure, and perceptual terms.

---

## 1. Project Directory Structure
```
team_name/
├── config.yaml          # Hyperparameters and directory mappings
├── requirements.txt     # Frozen Python dependencies with version details
├── README.md            # Repository documentation
├── run.py               # Mandatory evaluation entry point script
├── train.py             # Main PyTorch training & validation loop
├── src/
│   ├── dataset.py       # Custom dataset, robust normalization, and RAM caching
│   ├── losses.py        # Combined loss (Charbonnier + SSIM + LPIPS)
│   ├── metrices.py      # Evaluation metrics (PSNR, SSIM, LPIPS)
│   ├── model.py         # NAFNet-lite model definition
│   ├── split_dataset.py # Deterministic train/validation cluster split
│   └── synthetic_degrade.py # Online speckle noise and blur degradation models
├── models/
│   └── model_best.pt    # Best trained model checkpoint weights
└── outputs/
    ├── training_log.csv # CSV metrics log per epoch
    └── restored_test/   # Final restored test outputs
```

---

## 2. Setup & Installation

To run evaluation or training locally, set up the Python virtual environment and install the required dependencies:

```bash
# 1. Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install frozen requirements
pip install -r requirements.txt
```

---

## 3. Evaluation Guide (Submission Interface)

To run inference on a folder of test files, execute `run.py` using the following interface:

```bash
python run.py <input-dir> <output-dir>
```

### Key Script Features:
*   **Automatic Model Loading**: Automatically loads the best weights from `models/model_best.pt` inside the repository.
*   **Dynamic Format Matching**: Detects whether inputs are `.npy` arrays or standard images (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`) and saves outputs in the **exact same format**.
*   **No-Disk I/O RAM Cache**: Caches inputs directly in system memory during DataLoader access to bypass slow disk reads.
*   **Fast Batch Processing**: Scans PIL headers to group images by shape, executing inference in uniform batches for maximum GPU utilization.
*   **Precision Optimization**: Runs inference wrapped in `torch.no_grad()`, `model.eval()`, and utilizes **FP16 Autocast** (`torch.amp.autocast` / `torch.cuda.amp.autocast`) on CUDA and MPS GPUs.

---

## 4. Reproducing Training

To partition the dataset and run the 100-epoch training loop from scratch:

```bash
# Set PYTHONPATH to root directory
export PYTHONPATH=.

# 1. (Optional) Run the K-Means visual-style cluster splitter (85/15 split)
python3 src/split_dataset.py

# 2. Execute the PyTorch training loop
python3 train.py
```
*   *Note: Checkpoints will be saved epoch-by-epoch to `models/` and metrics will be logged to `outputs/training_log.csv`.*

---

## 5. Technical Design Details

### Model Architecture (`NAFNetLite`)
The model has **`2,303,041` trainable parameters** (budget: 2M – 8M). It consists of:
1.  **Shallow Feature Extractor**: Conv2d mapping the grayscale input channel to 96 baseline feature channels.
2.  **12 NAF Blocks**: Core restoration blocks containing LayerNorm2d, a **SimpleGate** channel-splitting multiplication replacing traditional GELU/ReLU activations, and **Simplified Channel Attention (SCA)**.
3.  **Learned 2x Upsampler**: A Convolutional layer expanding channels from 96 to 384, followed by a `PixelShuffle(2)` block to reconstruct 96 upscaled channels.
4.  **Reconstruction**: Conv2d projecting channels back to 1 grayscale output, wrapped in a `Sigmoid` activation to enforce strict `[0.0, 1.0]` bounds.

### Hybrid Loss Function ($L_{total}$)
We optimize a combined loss function targeted at pixel accuracy, structural preservation, and human visual perception:
$$L_{total} = 1.0 \times L_{Charbonnier} + 0.3 \times (1 - \text{SSIM}) + 0.05 \times L_{LPIPS}$$
*   **Charbonnier Loss**: A differentiable L1 approximation (with $\epsilon = 10^{-3}$) which is robust to outliers and noise.
*   **SSIM Loss**: Directly optimizes structural similarities in pixel groups.
*   **LPIPS Loss**: Perceptual quality loss leveraging AlexNet features to preserve edge sharpness.

### Robust Normalization
Clean ground-truth images are strictly in the range `[0.0, 1.0]`. However, due to speckle noise, low-resolution inputs contain significant overshoots ranging from `-0.278` to `2.158`.
*   Standard min-max normalization compresses the signal variance based on isolated noise spikes.
*   Instead, we use **Robust Normalization** mapping the dataset's 1st/99th percentiles (`[-0.15, 1.15]`) to `[0.0, 1.0]`, clipping any extreme remaining outliers. This preserves the local signal variance and prevents brightness bias in restored areas.
