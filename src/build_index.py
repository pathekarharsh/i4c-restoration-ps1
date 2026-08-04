import os
import csv
import numpy as np

def build_index():
    # Define paths relative to the repository root (i4c/)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    noisy_dir = os.path.join(repo_root, "data", "raw", "train", "NoisyLR")
    gt_dir = os.path.join(repo_root, "data", "raw", "train", "GT")
    output_csv = os.path.join(repo_root, "data", "train_pairs.csv")

    print(f"Project root: {repo_root}")
    print(f"Reading noisy inputs from: {noisy_dir}")
    print(f"Reading ground truth inputs from: {gt_dir}")

    if not os.path.exists(noisy_dir) or not os.path.exists(gt_dir):
        raise FileNotFoundError(
            f"Required directories not found! Make sure symlinks are created at {noisy_dir} and {gt_dir}"
        )

    noisy_files = sorted(os.listdir(noisy_dir))
    gt_files = sorted(os.listdir(gt_dir))

    # We assume filename-based matching as files are flat and numbered (000000.npy, 000001.npy, etc.)
    noisy_set = set(noisy_files)
    gt_set = set(gt_files)

    # Check for perfect alignment
    common_files = sorted(list(noisy_set.intersection(gt_set)))
    print(f"Found {len(common_files)} matching file pairs.")
    
    if len(common_files) != len(noisy_files) or len(common_files) != len(gt_files):
        print(f"Warning: Discrepancy in file count! Noisy: {len(noisy_files)}, GT: {len(gt_files)}, Matching: {len(common_files)}")

    pairs = []
    print("Verifying resolutions and indexing...")
    for idx, filename in enumerate(common_files):
        if not filename.endswith(".npy"):
            continue

        noisy_filepath = os.path.join(noisy_dir, filename)
        gt_filepath = os.path.join(gt_dir, filename)

        # Fast read of NumPy headers using memory mapping to avoid loading large arrays fully into RAM
        try:
            noisy_shape = np.load(noisy_filepath, mmap_mode="r").shape
            gt_shape = np.load(gt_filepath, mmap_mode="r").shape
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        resolution_pair = f"{noisy_shape[0]}->{gt_shape[0]}"
        
        # Check standard 2x super-resolution
        if noisy_shape[0] * 2 != gt_shape[0]:
            print(f"Warning: Unexpected resolution pair {resolution_pair} for file {filename}")

        # Paths should be relative to the project root for clean dataset loading later
        rel_noisy_path = os.path.relpath(noisy_filepath, repo_root)
        rel_gt_path = os.path.relpath(gt_filepath, repo_root)

        # Since the folder is flat and filenames are simple indices, source_id is set to 'default'.
        # We can analyze visual structures during EDA, but no metadata/folder structure defines source_id.
        source_id = "default"

        pairs.append({
            "degraded_path": rel_noisy_path,
            "gt_path": rel_gt_path,
            "resolution_pair": resolution_pair,
            "source_id": source_id
        })

    # Write to CSV file
    fieldnames = ["degraded_path", "gt_path", "resolution_pair", "source_id"]
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(pair)

    print(f"Successfully generated index csv at: {output_csv}")
    print(f"Total indexed pairs: {len(pairs)}")

if __name__ == "__main__":
    build_index()
