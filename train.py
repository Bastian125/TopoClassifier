"""
Train different ML models on given input features.
"""

# ---------- Imports ---------- #
import os
import argparse

import math
import numpy as np
import matplotlib.pyplot as plt
import h5py

import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from sklearn.model_selection import train_test_split

from config import data_save_path, output_path


# ---------- File Config ---------- #
feature_keys = [
    "clusterE",
    "cluster_FIRST_ENG_DENS",
    "cluster_EM_PROBABILITY",
    "cluster_CENTER_LAMBDA",
    "cluster_CENTER_MAG",
    "cluster_nCells_tot",
    "cluster_ENG_FRAC_EM",
    "cluster_SECOND_TIME",
    "cluster_AVG_TILE_Q",
    "cluster_AVG_LAR_Q",
    "cluster_SECOND_R",
    "cluster_LATERAL",
    "cluster_time",
    "cluster_ISOLATION",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 256
EPOCHS = 30
MODEL_OUTPUT = "trained_dnn_model.h5"


# ---------- Argument Parser ---------- #
parser = argparse.ArgumentParser(
    description="Train ML models on data of Run 2 and Run 3 of ATLAS."
)
data_group = parser.add_mutually_exclusive_group(required=True)
data_group.add_argument(
    "--mc20a",
    action="store_true",
    help="Train model with MC20a data for fast code tests.",
)
data_group.add_argument(
    "--mc20",
    action="store_true",
    help="Train model with MC20 data.",
)
data_group.add_argument(
    "--mc23",
    action="store_true",
    help="Train model with MC23 data.",
)
mode_group = parser.add_mutually_exclusive_group(required=True)
mode_group.add_argument(
    "--train",
    action="store_true",
    help="Train ML model.",
)
args = parser.parse_args()


# ---------- Helper Functions ---------- #
def load_data(filename):
    """
    Load hdf5 data for training and assign label = 1 for data w/o pile-up and label = 0 for data with pile-up.
    """
    file_path = os.path.join(data_save_path, filename)
    with h5py.File(file_path, "r") as f:
        X = np.stack([f[key][:] for key in feature_keys], axis=1)
        if "noPU" in os.path.basename(filename):
            y = np.ones(X.shape[0])
        elif "withPU" in os.path.basename(filename):
            y = np.zeros(X.shape[0])
    return X, y


def load_testrun_data():
    """
    Just load and concatenate MC20a with and w/o pile-up for code test runs.
    """
    print("Load data for test run...")
    X_bkg, y_bkg = load_data("mc20a_withPU_norm.h5")
    X_sig, y_sig = load_data("mc20a_noPU_norm.h5")
    X = np.concatenate([X_bkg, X_sig])
    y = np.concatenate([y_bkg, y_sig])
    return X, y


def load_full_data(campaign):
    """
    Load and concatenate all files with and w/o pile-up of one campaign (either 20 or 23).
    """
    print(f"Load full data for MC{campaign}...")
    if campaign == 20:
        X_bkg_a, y_bkg_a = load_data("mc20a_withPU_norm.h5")
        X_sig_a, y_sig_a = load_data("mc20a_noPU_norm.h5")
        X_bkg_d, y_bkg_d = load_data("mc20d_withPU_norm.h5")
        X_sig_d, y_sig_d = load_data("mc20d_noPU_norm.h5")
        X_bkg_e, y_bkg_e = load_data("mc20e_withPU_norm.h5")
        X_sig_e, y_sig_e = load_data("mc20e_noPU_norm.h5")
        X = np.concatenate([X_bkg_a, X_bkg_d, X_bkg_e, X_sig_a, X_sig_d, X_sig_e])
        y = np.concatenate([y_bkg_a, y_bkg_d, y_bkg_e, y_sig_a, y_sig_d, y_sig_e])
    if campaign == 23:
        X_bkg_a, y_bkg_a = load_data("mc23a_withPU_norm.h5")
        X_sig_a, y_sig_a = load_data("mc23a_noPU_norm.h5")
        X_bkg_d, y_bkg_d = load_data("mc23d_withPU_norm.h5")
        X_sig_d, y_sig_d = load_data("mc23d_noPU_norm.h5")
        X_bkg_e, y_bkg_e = load_data("mc23e_withPU_norm.h5")
        X_sig_e, y_sig_e = load_data("mc23e_noPU_norm.h5")
        X = np.concatenate([X_bkg_a, X_bkg_d, X_bkg_e, X_sig_a, X_sig_d, X_sig_e])
        y = np.concatenate([y_bkg_a, y_bkg_d, y_bkg_e, y_sig_a, y_sig_d, y_sig_e])
    return X, y


def build_dnn_model(input_dim):
    """
    Build DNN model for classifying hard-scatter and pile-up only clusters.
    """
    print("Build DNN model...")
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def plot_training_history(history):
    """
    Plot training history of ML model.
    """
    print("Plot training history...")
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "ML/training_history.pdf"))
    plt.close()


# ---------- Main Function ---------- #
def main():
    if args.train:
        # Load data
        if args.mc20a:
            X, y = load_testrun_data()
        elif args.mc20:
            X, y = load_data(20)
        elif args.mc23:
            X, y = load_data(23)

        # Apply train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        # Build and train DNN model
        dnn_model = build_dnn_model(X_train.shape[1])
        print("Train model...")
        history = dnn_model.fit(X_train, y_train, validation_split=0.35, epochs=EPOCHS, batch_size=BATCH_SIZE)

        # Save model
        dnn_model.save(os.path.join(output_path, MODEL_OUTPUT))
        print(f"Model saved to {os.path.join(output_path, MODEL_OUTPUT)}...")

        # Plot training history
        plot_training_history(history)


if __name__ == "__main__":
    main()
