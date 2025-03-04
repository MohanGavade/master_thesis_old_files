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

#  Paths
input_folder = r"/home/q674749/workspace/thesis_work/rat25-15.4.1/perception/std_DataSet"
output_folder = r"/home/q674749/workspace/thesis_work/rat25-15.4.1/perception/dbscan_results"
param_output_file = r"/home/q674749/workspace/thesis_work/rat25-15.4.1/perception/best_dbscan_params.csv"

os.makedirs(output_folder, exist_ok=True)

#  Define Clustering Features
clustering_features = ["distance", "radialVelocity", "rcs", 
                       "angleAzimuth_sin", "angleAzimuth_cos",
                       "angleElevation_sin", "angleElevation_cos"]

#  Grid Search Parameters
eps_values = [0.4, 1.0, 1.5, 2.0, 2.5]
min_samples_values = [3, 5, 7, 10, 12, 15]

#  Safe Memory Limit
MEMORY_THRESHOLD = 6 * 1024 * 1024 * 1024  # 6GB available RAM

#  Function to Check Memory Usage
def is_memory_safe():
    return psutil.virtual_memory().available > MEMORY_THRESHOLD  

#  Load Processed Files
def get_processed_files():
    if os.path.exists(param_output_file):
        df_params = pd.read_csv(param_output_file, header=None, names=["filename", "eps", "min_samples", "silhouette_score"])
        return set(df_params["filename"].tolist())
    return set()

#  Get Unprocessed Files
def get_files_to_process():
    all_files = set([f for f in os.listdir(input_folder) if f.endswith(".p")])
    processed_files = get_processed_files()
    return list(all_files - processed_files)

#  DBSCAN Processing Function (Grid Search Only)
def process_dbscan(filename):
    try:
        if not filename.endswith(".p"):  
            return None

        file_path = os.path.join(input_folder, filename)

        if os.path.getsize(file_path) == 0:
            print(f"⚠ [WARNING] Empty file: {filename}. Skipping.")
            return None

        df = pd.read_pickle(file_path)
        print(f"\n🔄 [INFO] Processing File: {filename} ...")

        X_sample = []
        for _, row in df.iterrows():
            for i in range(len(row["distance"])):
                X_sample.append([row[col][i] for col in clustering_features])

        if not X_sample:
            print(f"⚠ [WARNING] No valid data found in {filename}. Skipping.")
            return None

        X_sample = np.array(X_sample)

        #  Store Best Grid Search Results
        best_params = (1.0, 5)
        best_score = -1

        print(f"\n🚀 [INFO] Running Grid Search for {filename} ...")
        for eps, min_samples in tqdm(product(eps_values, min_samples_values), desc=f"Grid {filename}", leave=False):
            if not is_memory_safe():  
                print(f"⚠ [WARNING] Low memory! Stopping Grid Search for {filename}.")
                break

            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=1)
            labels = dbscan.fit_predict(X_sample)

            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            score = silhouette_score(X_sample, labels) if num_clusters > 1 else -1

            if score > best_score:
                best_score = score
                best_params = (eps, min_samples)

        print(f"\n✅ [INFO] Best Grid Search Params for {filename}: eps={best_params[0]}, min_samples={best_params[1]}")

        #  Save Best Parameters to CSV (Efficient)
        with open(param_output_file, "a") as f:
            f.write(f"{filename},{best_params[0]},{best_params[1]},{best_score}\n")

        gc.collect()
        return filename, best_params

    except Exception as e:
        print(f"⚠ [ERROR] Failed to process {filename}: {str(e)}")
        return None


#  Run Multi-Core Processing for Grid Search on Unprocessed Files
if __name__ == "__main__":
    files_to_process = get_files_to_process()

    if len(files_to_process) == 0:
        print("\n✅ [INFO] All files are already processed. No new files to process.")
    else:
        print(f"\n🚀 [INFO] Starting Multi-Core Grid Search on {len(files_to_process)} missing files...")

        with multiprocessing.Pool(processes=4) as pool:
            list(tqdm(pool.imap(process_dbscan, files_to_process), total=len(files_to_process), desc="Processing Skipped Files"))

        print(f"\n✅ [INFO] Grid Search Completed for Remaining Files! Updated {param_output_file}")