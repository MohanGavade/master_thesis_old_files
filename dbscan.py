import os
import gc
import time
import psutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import multiprocessing

# ✅ Paths
input_folder = r"/home/q674749/workspace/thesis_work/rat25-15.4.1/perception/bappa"
output_folder = r"/home/q674749/workspace/thesis_work/rat25-15.4.1/perception/dbscan_results"
param_output_file = r"/home/q674749/workspace/thesis_work/rat25-15.4.1/perception/best_dbscan_params.csv"
skipped_files_log = r"/home/q674749/workspace/thesis_work/rat25-15.4.1/perception/skipped_files.txt"

os.makedirs(output_folder, exist_ok=True)

# ✅ Define Clustering Features
clustering_features = ["distance", "radialVelocity", "rcs", 
                       "angleAzimuth_sin", "angleAzimuth_cos",
                       "angleElevation_sin", "angleElevation_cos"]

# ✅ Grid Search Parameters
eps_values = [0.5, 1.0, 1.5, 2.0, 2.5]
min_samples_values = [3, 5, 7, 10, 12, 15]

# ✅ Safe Memory Limit
MEMORY_THRESHOLD = 6 * 1024 * 1024 * 1024  # Stop processing if available RAM < 6GB

# ✅ Initial Chunk Size
CHUNK_SIZE = 500  # Will reduce dynamically if needed

# ✅ Initial Number of Parallel Processes
NUM_PROCESSES = 4  # Will reduce dynamically if memory runs low

# ✅ Function to Check Memory Usage
def is_memory_safe():
    return psutil.virtual_memory().available > MEMORY_THRESHOLD  

# ✅ Load Processed Files to Avoid Redundant Processing
def get_processed_files():
    if os.path.exists(param_output_file):
        df_params = pd.read_csv(param_output_file, header=None, names=["filename", "eps", "min_samples", "silhouette_score"])
        return set(df_params["filename"].tolist())
    return set()

# ✅ Get Skipped Files from Log
def get_skipped_files():
    if os.path.exists(skipped_files_log):
        with open(skipped_files_log, "r") as f:
            return set(f.read().splitlines())
    return set()

# ✅ Get Files That Still Need Processing
def get_files_to_process():
    all_files = set([f for f in os.listdir(input_folder) if f.endswith(".p")])
    processed_files = get_processed_files()
    skipped_files = get_skipped_files()
    return list((all_files - processed_files) | skipped_files)

# ✅ DBSCAN Processing Function (Fixed for Variable-Length Inputs)
def process_dbscan(filename):
    global CHUNK_SIZE

    try:
        if not filename.endswith(".p"):  
            return None

        file_path = os.path.join(input_folder, filename)

        if os.path.getsize(file_path) == 0:
            print(f"⚠ [WARNING] Empty file: {filename}. Skipping.")
            return None

        df = pd.read_pickle(file_path)
        print(f"\n🔄 [INFO] Processing File: {filename} ...")

        # ✅ Flatten and Convert Data to Ensure Proper Input Format
        X_sample = []
        for _, row in df.iterrows():
            num_points = len(row["distance"])

            # Process in smaller chunks to save memory
            for i in range(0, num_points, CHUNK_SIZE):
                chunk = [row[col][i:i + CHUNK_SIZE] for col in clustering_features]

                # Ensure all elements are lists and same length
                if all(len(x) == len(chunk[0]) for x in chunk):
                    X_sample.extend(np.array(chunk).T.tolist())  # Convert to row-wise format

        if not X_sample:
            print(f"⚠ [WARNING] No valid data found in {filename}. Skipping.")
            return None

        X_sample = np.array(X_sample)

        # ✅ Store Best Grid Search Results
        best_params = (1.0, 5)
        best_score = -1

        print(f"\n🚀 [INFO] Running Grid Search for {filename} ...")
        for eps, min_samples in tqdm(product(eps_values, min_samples_values), desc=f"Grid {filename}", leave=False):
            if not is_memory_safe():  
                print(f"⚠ [WARNING] Low memory! Reducing chunk size and processes.")
                CHUNK_SIZE = max(250, CHUNK_SIZE // 2)  # Dynamically reduce chunk size
                return None  

            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=1)
            labels = dbscan.fit_predict(X_sample)

            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            score = silhouette_score(X_sample, labels) if num_clusters > 1 else -1

            if score > best_score:
                best_score = score
                best_params = (eps, min_samples)

        print(f"\n✅ [INFO] Best Grid Search Params for {filename}: eps={best_params[0]}, min_samples={best_params[1]}")

        # ✅ Save Best Parameters to CSV
        with open(param_output_file, "a") as f:
            f.write(f"{filename},{best_params[0]},{best_params[1]},{best_score}\n")

        gc.collect()
        return filename, best_params

    except Exception as e:
        print(f"⚠ [ERROR] Failed to process {filename}: {str(e)}")
        return None


# ✅ Run Multi-Core Processing for Grid Search on Unprocessed Files
if __name__ == "__main__":
    files_to_process = get_files_to_process()

    if len(files_to_process) == 0:
        print("\n✅ [INFO] All files are already processed. No new files to process.")
    else:
        print(f"\n🚀 [INFO] Starting Multi-Core Grid Search on {len(files_to_process)} missing files...")

        with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
            list(tqdm(pool.imap(process_dbscan, files_to_process), total=len(files_to_process), desc="Processing Skipped Files"))

        print(f"\n✅ [INFO] Grid Search Completed for Remaining Files! Updated {param_output_file}")

    # ✅ Process Skipped Files with Lower Memory Usage
    skipped_files = get_skipped_files()
    if skipped_files:
        print(f"\n⚠ [WARNING] {len(skipped_files)} files were skipped due to low memory. Retrying with reduced chunk size and parallelism...")

        CHUNK_SIZE = 250  # Reduce chunk size further
        NUM_PROCESSES = 2  # Reduce parallelism

        with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
            list(tqdm(pool.imap(process_dbscan, list(skipped_files)), total=len(skipped_files), desc="Reprocessing Skipped Files"))

        print("\n✅ [INFO] Skipped files reprocessed with lower memory usage.")