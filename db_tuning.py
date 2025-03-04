import os
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ✅ Paths
input_folder = r"/home/q674749/workspace/thesis_work/rat25-15.4.1/perception/tunning_data"
output_folder = os.path.join(input_folder, "dbscan_results")
param_output_file = os.path.join(input_folder, "best_dbscan_params.csv")

os.makedirs(output_folder, exist_ok=True)

# ✅ Clustering Features
clustering_features = ["distance", "radialVelocity", "rcs", 
                       "angleAzimuth_sin", "angleAzimuth_cos",
                       "angleElevation_sin", "angleElevation_cos"]

# ✅ Grid Search Parameters
eps_values = [0.5, 1.0, 1.5, 2.0, 2.5]  # ✅ Try different neighborhood distances
min_samples_values = [3, 5, 7, 10, 12, 15]  # ✅ Try different cluster sizes

# ✅ Sampling Config (Processing 50% of Full Rows)
row_sampling_ratio = 0.50  # ✅ Use 50% of rows for each parameter set

# ✅ Retrieve Sample Files
def get_sample_files():
    """Retrieve 4 sample .p files from the input folder."""
    all_files = [f for f in os.listdir(input_folder) if f.endswith(".p")]
    return sorted(all_files)[:4]  # Pick the first 4 files in sorted order

# ✅ Evaluation Function (Weighted Scoring)
def evaluate_clustering(X, labels):
    """Compute evaluation metrics for DBSCAN clustering."""
    try:
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_percentage = np.sum(labels == -1) / len(labels)

        # Silhouette Score
        sil_score = silhouette_score(X, labels) if num_clusters > 1 else -1

        # Davies-Bouldin Index
        dbi_score = davies_bouldin_score(X, labels) if num_clusters > 1 else float("inf")

        # ✅ Weighted Final Score
        final_score = sil_score - (dbi_score * 0.1) - (noise_percentage * 0.1)

        return final_score

    except Exception as e:
        return -1  # Default worst case if error

# ✅ Run DBSCAN Grid Search
def run_dbscan_grid_search(filename):
    """Run DBSCAN tuning using Grid Search."""
    file_path = os.path.join(input_folder, filename)
    
    if not os.path.exists(file_path):
        print(f"⚠ [WARNING] File {filename} not found. Skipping.")
        return None

    df = pd.read_pickle(file_path)
    num_rows = len(df)
    print(f"\n🔄 [INFO] Processing File: {filename} ...")
    print(f"📊 [INFO] Total frames in {filename}: {num_rows}")

    # ✅ Select the same 50% of rows for tuning this file
    selected_rows = np.random.choice(num_rows, size=int(num_rows * row_sampling_ratio), replace=False)
    selected_rows.sort()  # Sort for ordered processing

    print(f"📊 [INFO] Using {len(selected_rows)} frames (50% sample) for tuning.")

    best_params = None
    best_score = -float("inf")

    # ✅ Iterate over DBSCAN parameters with a progress bar
    for eps, min_samples in tqdm(product(eps_values, min_samples_values), total=len(eps_values) * len(min_samples_values), desc=f"Tuning {filename}"):

        print(f"\n🚀 [INFO] Trying eps={eps}, min_samples={min_samples} on {filename}...")

        scores = []
        # ✅ Process the same selected rows for each parameter set
        for row_index in selected_rows:
            row = df.iloc[row_index]
            X_sample = np.array([[row[col][i] for col in clustering_features] for i in range(len(row["distance"]))])

            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=1)
            labels = dbscan.fit_predict(X_sample)

            final_score = evaluate_clustering(X_sample, labels)
            scores.append(final_score)

        # ✅ Compute average score for the file
        avg_score = np.mean(scores)

        # ✅ Store best parameters
        if avg_score > best_score:
            best_score = avg_score
            best_params = (eps, min_samples)

        gc.collect()  # ✅ Free up memory

    # ✅ Find and Print Best Parameters (Fix NoneType Error)
    if best_params:
        best_eps, best_min_samples = best_params
        print(f"\n✅ [BEST PARAMS] {filename} → eps={best_eps}, min_samples={best_min_samples}, Score: {best_score:.3f}")
    else:
        print(f"\n❌ [ERROR] No valid clusters found for {filename}. Defaulting to eps=1.0, min_samples=3.")
        best_eps, best_min_samples, best_score = 1.0, 3, -1  # Default values

    return filename, best_eps, best_min_samples, best_score

# ✅ Run for All Files
sample_files = get_sample_files()
best_results = {}

if sample_files:
    print(f"\n🚀 [INFO] Starting DBSCAN tuning on {len(sample_files)} sample files...")
    for file in sample_files:
        filename, best_eps, best_min_samples, best_score = run_dbscan_grid_search(file)
        best_results[filename] = (best_eps, best_min_samples, best_score)

    # ✅ Print Final Best Parameters for All Files
    print("\n🚀 *Final Best Parameters for Each File:*")
    for file, (eps, min_samples, score) in best_results.items():
        print(f"📌 {file} → eps={eps}, min_samples={min_samples}, Score: {score:.3f}")

else:
    print("\n⚠ [ERROR] No valid .p files found in the tuning data folder!")