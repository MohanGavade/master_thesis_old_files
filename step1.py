import os
import pandas as pd

# Define new folder paths
base_folder = "/home/q674749/workspace/thesis_work/rat25-15.4.1"
detections_folder = os.path.join(base_folder, "frr40_detections")
objects_folder = os.path.join(base_folder, "frr40_objects")
ego_motion_folder = os.path.join(base_folder, "egoMotionDynamicData")
processed_folder = os.path.join(base_folder, "processed")  # For saving new processed files
os.makedirs(processed_folder, exist_ok=True)

# Columns to retain
detections_columns = [
    "timestamp", "rcs", "distance", "angleAzimuth", "angleElevation", "radialVelocity", "radialVelocityDomainMax"
]
objects_columns = [
    "timestamp", "orientation", "x", "y", "width_edge_mean", "length_edge_mean",
    "status_measurement", "status_movement", "overdrivable", "underdrivable",
    "header.origin.x", "header.origin.y", "header.origin.z",
    "header.origin.roll", "header.origin.pitch", "header.origin.yaw", "reference_point"
]
ego_motion_columns = [
    "timestamp", "RotationRates.yawRateVehicleBody.value", "Velocity.SpeedCog.SpeedCog"
]

# Function to filter files
def filter_files(folder, columns_to_keep, output_subfolder):
    output_path = os.path.join(processed_folder, output_subfolder)
    os.makedirs(output_path, exist_ok=True)
    filtered_data = {}

    files = sorted([f for f in os.listdir(folder) if f.endswith(".p")])
    for file in files:
        file_path = os.path.join(folder, file)
        print(f"Processing file: {file}")
        
        # Load data
        data = pd.read_pickle(file_path)
        
        # Filter relevant columns
        filtered_df = data[columns_to_keep]
        filtered_data[file] = filtered_df
        
        # Save filtered file
        save_path = os.path.join(output_path, file)
        filtered_df.to_pickle(save_path)
        print(f"Filtered data saved to: {save_path}")

    return filtered_data

# Filter detections, objects, and ego motion files
filtered_detections = filter_files(detections_folder, detections_columns, "filtered_detections")
filtered_objects = filter_files(objects_folder, objects_columns, "filtered_objects")
filtered_ego_motion = filter_files(ego_motion_folder, ego_motion_columns, "filtered_ego_motion")

# Verify the results
print("\nFiltered files are saved in the 'processed' folder.")
