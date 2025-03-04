import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

#  Paths
train_folder = "/home/q674749/workspace/thesis_work/rat25-15.4.1/perception/train_data"

#  Training Parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

#  Feature & Label Definitions
feature_columns = ["rcs", "distance", "angleAzimuth", "angleElevation", "radialVelocity"]
scalar_columns = ["radialVelocityDomainMax", "yaw_rate", "ego_speed"]
object_properties = ["centroid_x", "centroid_y", "width_edge_mean", "length_edge_mean", "orientation"]

#  Load Data from Pickle Files
def load_radar_data(data_folder):
    files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".p")]

    if not files:
        raise ValueError(f" [ERROR] No valid .p files found in {data_folder}")

    data_list = []
    for file_path in tqdm(files, desc="Loading Train Files"):
        df = pd.read_pickle(file_path)
        if df.empty:
            print(f" [WARNING] Empty DataFrame in file: {file_path}")
            continue
        data_list.append(df)

    if not data_list:
        raise ValueError(" [ERROR] No valid data loaded from pickle files.")

    return pd.concat(data_list, ignore_index=True)

#  Load and Process Training Data
print("\n [INFO] Loading Training Dataset...")
try:
    train_df = load_radar_data(train_folder)
    print(f" [INFO] Successfully loaded {len(train_df)} samples.")
except Exception as e:
    print(f" [ERROR] Failed to load dataset: {e}")
    exit()

# ✅ Convert DataFrame to NumPy Arrays
def preprocess_data(df):
    features = np.stack(df["features"].values)  # Convert list of lists to NumPy array
    scalars = np.stack(df["scalars"].values)
    labels = np.stack(df["labels"].values)

    #  Apply Mask (Ensure masked labels are ignored)
    if "mask" in df.columns:
        mask = np.stack(df["mask"].values)
    else:
        print("\n [WARNING] 'mask' column is missing! Using default mask (all ones).")
        mask = np.ones_like(labels)

    return features, scalars, labels, mask

X_features, X_scalars, y_labels, y_mask = preprocess_data(train_df)

#  Create TensorFlow Dataset
def create_tf_dataset(X_features, X_scalars, y_labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(({"features": X_features, "scalars": X_scalars}, y_labels))
    dataset = dataset.shuffle(buffer_size=len(X_features)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_tf_dataset(X_features, X_scalars, y_labels, BATCH_SIZE)

#  Define Simple Feedforward Model in TensorFlow
input_features = keras.Input(shape=(len(feature_columns),), name="features")
input_scalars = keras.Input(shape=(len(scalar_columns),), name="scalars")

# Concatenate Inputs
x = layers.Concatenate()([input_features, input_scalars])
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)
output = layers.Dense(len(object_properties), activation="linear")(x)  # Regression task

#  Create Model
model = keras.Model(inputs=[input_features, input_scalars], outputs=output)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss="mse", metrics=["mae"])

#  Train Model
print("\n [INFO] Training Model...")
history = model.fit(train_dataset, epochs=EPOCHS, verbose=1)

#  Save Model
model.save("radar_object_predictor_tf.keras")
print("\n [SAVED] Model Saved Successfully!")

#  Plot Training Performance
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training Loss (MSE)")
plt.plot(history.history["mae"], label="Mean Absolute Error")
plt.xlabel("Epochs")
plt.ylabel("Loss / Error")
plt.legend()
plt.title("Training Performance")
plt.show()