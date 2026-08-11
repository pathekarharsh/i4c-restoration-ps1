import os
import argparse
import time
import torch
import numpy as np
from PIL import Image
from src.dataset import normalize, denormalize
from src.model import NAFNetLite

def get_shape(fpath):
    """Reads the array/image shape quickly without loading full pixel data in memory."""
    ext = os.path.splitext(fpath)[1].lower()
    if ext == ".npy":
        return np.load(fpath, mmap_mode="r").shape
    else:
        with Image.open(fpath) as img:
            # PIL returns (width, height), we return (height, width)
            return (img.height, img.width)

def load_image_or_array(fpath):
    """Loads array or image file, returns normalized float32 array in [0.0, 1.0]."""
    ext = os.path.splitext(fpath)[1].lower()
    if ext == ".npy":
        arr = np.load(fpath).astype(np.float32)
        return normalize(arr), ext
    else:
        # Open standard image as grayscale
        with Image.open(fpath).convert("L") as img:
            arr = np.array(img).astype(np.float32) / 255.0
            return arr, ext

def save_image_or_array(out_path, restored, ext):
    """Saves output to matching original format (.npy or standard image)."""
    if ext == ".npy":
        np.save(out_path, restored)
    else:
        # Scale to [0, 255] and save as standard image file
        img_out = (np.clip(restored, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        img = Image.fromarray(img_out)
        img.save(out_path)

def main():
    parser = argparse.ArgumentParser(description="Evaluate KLA Image Restoration Model")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing test images/arrays")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save restored outputs")
    parser.add_argument("positional_args", nargs="*", help="Positional arguments: [input_dir, output_dir]")
    args = parser.parse_args()

    # 1. Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using Device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Device: MPS")
    else:
        device = torch.device("cpu")
        print("Using Device: CPU")

    # 2. Paths
    repo_root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(repo_root, "weights", "model_best.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model checkpoint not found at: {model_path}")

    # 3. Initialize and load model
    config_path = os.path.join(repo_root, "config.yaml")
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = NAFNetLite(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        num_features=config["model"]["num_features"],
        num_blocks=config["model"]["num_blocks"]
    )
    
    print(f"Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Setup FP16 Autocast Context for GPU speed benchmarking
    device_type = device.type
    class DummyContext:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc_val, exc_tb): return None

    if device_type == "cuda":
        autocast_ctx = torch.amp.autocast(device_type="cuda")
        print("Using FP16 Autocast Inference (CUDA).")
    elif device_type == "mps":
        try:
            autocast_ctx = torch.amp.autocast(device_type="mps")
            print("Using FP16 Autocast Inference (MPS).")
        except:
            autocast_ctx = DummyContext()
            print("MPS Autocast context not supported in this torch version, using FP32.")
    else:
        autocast_ctx = DummyContext()
        print("Using CPU FP32 Inference.")

    # 4. Read inputs and group them by shape to batch process efficiently
    if args.input_dir and args.output_dir:
        input_dir = args.input_dir
        output_dir = args.output_dir
    elif len(args.positional_args) >= 2:
        input_dir = args.positional_args[0]
        output_dir = args.positional_args[1]
    else:
        parser.print_help()
        print("\nError: Please provide both input_dir and output_dir, either as flags (--input_dir / --output_dir) or as positional arguments.")
        return

    os.makedirs(output_dir, exist_ok=True)

    supported_exts = (".npy", ".png", ".jpg", ".jpeg", ".tiff", ".tif")
    filenames = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(supported_exts)])
    if not filenames:
        print(f"No supported files ({supported_exts}) found in {input_dir}!")
        return

    print(f"Found {len(filenames)} files in input directory.")

    # Group files by shape for clean batch processing
    shape_groups = {}
    for fname in filenames:
        fpath = os.path.join(input_dir, fname)
        shape = get_shape(fpath)
        shape_groups.setdefault(shape, []).append(fname)

    # 5. Run inference with timing
    start_time = time.time()
    total_images = len(filenames)
    processed_count = 0

    batch_size = config["training"].get("batch_size", 16)

    print("\nStarting batch inference...")
    with torch.no_grad():
        for shape, files in shape_groups.items():
            print(f"  Processing group with shape {shape} ({len(files)} files)...")
            
            # Batch loop
            for i in range(0, len(files), batch_size):
                batch_files = files[i:i + batch_size]
                
                # Load and stack arrays/images
                batch_data = []
                batch_exts = []
                for fname in batch_files:
                    fpath = os.path.join(input_dir, fname)
                    arr, ext = load_image_or_array(fpath)
                    batch_data.append(arr)
                    batch_exts.append(ext)
                
                # Convert to tensor: shape (B, 1, H, W)
                inputs = torch.tensor(np.array(batch_data), dtype=torch.float32).unsqueeze(1).to(device)
                
                # Model inference with FP16 autocasting
                with autocast_ctx:
                    outputs = model(inputs)
                
                # Transfer back to CPU and convert back to numpy
                outputs_cpu = outputs.cpu().numpy()
                
                # Save each file in batch
                for idx, fname in enumerate(batch_files):
                    restored_norm = outputs_cpu[idx, 0] # shape (H, W)
                    ext = batch_exts[idx]
                    
                    if ext == ".npy":
                        # Denormalize back to the original physical scale for arrays
                        restored = denormalize(restored_norm)
                    else:
                        # Standard image files are already on [0.0, 1.0] scale
                        restored = restored_norm
                    
                    # Strictly clamp final outputs to target [0.0, 1.0] range
                    restored = np.clip(restored, 0.0, 1.0)
                    
                    out_path = os.path.join(output_dir, fname)
                    save_image_or_array(out_path, restored, ext)
                    
                processed_count += len(batch_files)
                print(f"    Progress: {processed_count}/{total_images} images saved.", end="\r")

    end_time = time.time()
    total_time = end_time - start_time
    avg_time = (total_time / total_images) * 1000  # ms per image

    print(f"\nInference completed in {total_time:.2f} seconds.")
    print(f"Average time per image: {avg_time:.2f} ms")
    print(f"Restored outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()
