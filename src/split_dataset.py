import os
import csv
import numpy as np

def split_dataset():
    # Define paths relative to the repository root (i4c/)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv = os.path.join(repo_root, "data", "train_pairs.csv")
    train_output_csv = os.path.join(repo_root, "data", "train_pairs.csv") # We will overwrite this with the 85% training split
    val_output_csv = os.path.join(repo_root, "data", "val_pairs.csv")

    print(f"Reading index from: {input_csv}")
    
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Index CSV not found at {input_csv}. Please run build_index.py first.")

    # Load all indexed pairs
    pairs = []
    with open(input_csv, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pairs.append(row)

    print(f"Loaded {len(pairs)} image pairs from index.")

    # Extract features from GT images for clustering
    print("Extracting image features for clustering...")
    features = []
    for idx, pair in enumerate(pairs):
        gt_path = os.path.join(repo_root, pair["gt_path"])
        try:
            arr = np.load(gt_path)
            mean_val = arr.mean()
            std_val = arr.std()
            # Average spatial gradient magnitude (measure of edge density / texture complexity)
            dy, dx = np.gradient(arr)
            grad_mag = np.mean(np.sqrt(dx**2 + dy**2))
            features.append([mean_val, std_val, grad_mag])
        except Exception as e:
            print(f"Error loading {pair['gt_path']}: {e}")
            # Fallback to zero features in case of error
            features.append([0.5, 0.2, 0.1])

    features = np.array(features)

    # Normalize features for clustering
    feat_min = features.min(axis=0)
    feat_max = features.max(axis=0)
    norm_features = (features - feat_min) / (feat_max - feat_min + 1e-8)

    # Perform simple, deterministic K-Means clustering (K=3)
    # We choose three initial centroids deterministically for reproducibility
    print("Clustering images into 3 visual types...")
    n_samples = len(norm_features)
    centroids = norm_features[[0, n_samples // 3, 2 * n_samples // 3]].copy()
    
    labels = np.zeros(n_samples, dtype=int)
    for epoch in range(50):
        # Compute distances to centroids
        dists = np.linalg.norm(norm_features[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        
        # Recompute centroids
        new_centroids = np.zeros_like(centroids)
        for c in range(3):
            cluster_points = norm_features[new_labels == c]
            if len(cluster_points) > 0:
                new_centroids[c] = cluster_points.mean(axis=0)
            else:
                new_centroids[c] = centroids[c] # keep old centroid if empty
                
        if np.allclose(centroids, new_centroids) and np.array_equal(labels, new_labels):
            break
        centroids = new_centroids
        labels = new_labels

    for c in range(3):
        print(f"  Cluster {c}: {np.sum(labels == c)} images")

    # Update source_id in our list of pairs based on cluster labels
    for idx, pair in enumerate(pairs):
        pair["source_id"] = f"cluster_{labels[idx]}"

    # Identify cluster sizes and select one for validation split
    # Let's check which cluster has at least 480 items to hold out as OOD validation split.
    # We will hold out exactly 480 files (15% of 3200) from Cluster 2.
    val_indices = []
    cluster_2_indices = np.where(labels == 2)[0]
    
    if len(cluster_2_indices) >= 480:
        # Take exactly the first 480 images from Cluster 2 for the validation split
        np.random.seed(42)
        val_indices = np.random.choice(cluster_2_indices, size=480, replace=False)
    else:
        # Fallback if cluster 2 is somehow too small, take from cluster 1
        print("Warning: Cluster 2 was smaller than 480 images! Splitting across clusters instead.")
        # Fallback to random 15% split
        np.random.seed(42)
        val_indices = np.random.choice(n_samples, size=480, replace=False)

    val_indices_set = set(val_indices)
    
    train_pairs = []
    val_pairs = []
    
    for idx, pair in enumerate(pairs):
        if idx in val_indices_set:
            val_pairs.append(pair)
        else:
            train_pairs.append(pair)

    print(f"Splitting done: {len(train_pairs)} training pairs (85%), {len(val_pairs)} validation pairs (15%).")
    
    # Save training pairs to train_pairs.csv
    fieldnames = ["degraded_path", "gt_path", "resolution_pair", "source_id"]
    with open(train_output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for pair in train_pairs:
            writer.writerow(pair)
            
    # Save validation pairs to val_pairs.csv
    with open(val_output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for pair in val_pairs:
            writer.writerow(pair)

    print(f"Overwrote training index: {train_output_csv}")
    print(f"Generated validation index: {val_output_csv}")

if __name__ == "__main__":
    split_dataset()
