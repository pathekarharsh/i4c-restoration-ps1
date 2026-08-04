# KLA Image Restoration - NAFNet-lite Pipeline

This repository implements a lightweight, high-performance image restoration (denoising + 2x super-resolution) pipeline built for the KLA Image Restoration Hackathon.

## Repository Layout
```
i4c/
├── config.yaml          # Hyperparameters and directory mappings
├── requirements.txt     # Frozen Python dependencies
├── evaluate.py          # CLI script for evaluating test arrays
├── train.py             # Main PyTorch training & validation script
├── src/
│   ├── dataset.py       # Custom PyTorch Dataset & normalization helpers
│   ├── losses.py        # Hybrid loss definition (Charbonnier + SSIM + LPIPS)
│   ├── metrices.py      # Validation metrics (PSNR, SSIM, LPIPS)
│   ├── model.py         # NAFNet-lite restoration model
│   ├── split_dataset.py # Deterministic style-clustering dataset split
│   └── synthetic_degrade.py # Multiplicative speckle and blur degradation models
├── data/
│   ├── raw/             # Symlinks pointing to the raw npy arrays (gitignored)
│   ├── train_pairs.csv  # 85% training split index
│   └── val_pairs.csv    # 15% validation split index
├── weights/             # Saved checkpoints (best model stored as model_best.pt)
└── outputs/             # CSV training log and progress images
```

## Setup Instructions

1. **Virtual Environment Setup**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Scaffold Data Directory**:
   Run the symlink scaffolding commands to hook up the raw folders inside `data/raw/`.

## Running the Training Loop
To partition the dataset and execute the PyTorch training loop:
```bash
# 1. Split the indexed pairs into train and val subsets (85/15 ratio)
export PYTHONPATH=.
python3 src/split_dataset.py

# 2. Run the main training loop (saves logs and plots to outputs/ and weights to weights/)
python3 train.py
```

## Running Evaluation
To run inference on a folder of test arrays:
```bash
python evaluate.py --input_dir /path/to/test/images --output_dir /path/to/save
```
Outputs are automatically restored, denormalized, clipped to the target range `[0.0, 1.0]`, and saved with matching filenames in the output directory.
