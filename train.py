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
from sklearn.utils import class_weight

from config import data_save_path, output_path
from io_utils import ensure_dir_exists
from dataloader import HDF5DataGenerator


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
BATCH_SIZE = 512
EPOCHS = 3


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

model_group = parser.add_mutually_exclusive_group(required=True)
model_group.add_argument(
    "--DNN1",
    action="store_true",
    help="DNN model that classifies hard-scatter and pile-up clusters.",
)
model_group.add_argument(
    "--DNN2",
    action="store_true",
    help="DNN model that classifies pile-up only and mixed clusters.",
)

mode_group = parser.add_mutually_exclusive_group(required=True)
mode_group.add_argument(
    "--train",
    action="store_true",
    help="Train ML model.",
)
mode_group.add_argument(
    "--test",
    action="store_true",
    help="Test ML model.",
)
mode_group.add_argument(
    "--plot", action="store_true", help="Plots the training history."
)
args = parser.parse_args()


# --------- Define Dataset and Model Type --------- #
if args.mc20a:
    dataset_str = "Run2a"
elif args.mc20:
    dataset_str = "Run2"
elif args.mc23:
    dataset_str = "Run3"
else:
    raise ValueError("No dataset selected")

if args.DNN1:
    model_str = "DNN1"
elif args.DNN2:
    model_str = "DNN2"
else:
    raise ValueError("No model type selected")


# ---------- Helper Functions ---------- #
def load_data(filename):
    """
    Load hdf5 data for training and assign label = 1 for data w/o pile-up and label = 0 for data with pile-up.
    """
    file_path = os.path.join(data_save_path, filename)
    with h5py.File(file_path, "r") as f:
        X = np.stack([f[key][:] for key in feature_keys], axis=1)
        y = f["label"][:]
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


def build_dnn_model(input_dim, lr):
    """
    Build DNN model for classifying hard-scatter and pile-up only clusters.
    """
    print("Build DNN model...")
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(8, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def plot_training_history(history):
    """
    Plot training history of ML model.
    """
    print("Plot training history...")
    fig = plt.figure(figsize=[12, 6])
    gs = fig.add_gridspec(2, hspace=0.1, height_ratios=[1, 0.3])
    ax = gs.subplots(sharex=True, sharey=False)
    ax[0].plot(
        history.history["loss"],
        "bo",
        label="loss",
        markersize=1.5,
        linestyle="dashed",
    )
    ax[0].plot(
        history.history["val_loss"],
        "go",
        label="val_loss",
        markersize=1.5,
        linestyle="dashed",
    )
    ax[1].plot(
        np.array(history.history["val_loss"]) - np.array(history.history["loss"]),
        "bo",
        markersize=2,
        linestyle="dashed",
        label="val_loss - train_loss",
    )

    ax[1].set_xlabel("Epoch")
    ax[1].set_ylim(-0.03, 0.03)
    # Show only ticks and labels in the outer sides of the plots
    for a in ax:
        a.label_outer()
    ax[0].legend()
    ax[1].legend()
    ax[0].grid(True)
    ax[1].grid(True)
    output_directory = os.path.join(output_path, "ML", dataset_str)
    ensure_dir_exists(output_directory)
    save_path = os.path.join(output_directory, f"{model_str}_training_history.pdf")
    plt.savefig(save_path)
    plt.close()


# ---------- Main Function ---------- #
def main():
    # Prepare file paths
    if args.mc20a:
        file_paths = [
            os.path.join(data_save_path, "mc20a_withPU_norm.h5"),
            os.path.join(data_save_path, "mc20a_noPU_norm.h5"),
        ]
    elif args.mc20:
        file_paths = [
            os.path.join(data_save_path, "mc20a_withPU_norm.h5"),
            os.path.join(data_save_path, "mc20a_noPU_norm.h5"),
            os.path.join(data_save_path, "mc20d_withPU_norm.h5"),
            os.path.join(data_save_path, "mc20d_noPU_norm.h5"),
            os.path.join(data_save_path, "mc20e_withPU_norm.h5"),
            os.path.join(data_save_path, "mc20e_noPU_norm.h5"),
        ]
    elif args.mc23:
        file_paths = [
            os.path.join(data_save_path, "mc23a_withPU_norm.h5"),
            os.path.join(data_save_path, "mc23a_noPU_norm.h5"),
            os.path.join(data_save_path, "mc23d_withPU_norm.h5"),
            os.path.join(data_save_path, "mc23d_noPU_norm.h5"),
            os.path.join(data_save_path, "mc23e_withPU_norm.h5"),
            os.path.join(data_save_path, "mc23e_noPU_norm.h5"),
        ]
    else:
        raise ValueError("No dataset selected")


    if args.train:
        # Create generators
        train_generator = HDF5DataGenerator(
            file_paths=file_paths,
            feature_keys=feature_keys,
            batch_size=BATCH_SIZE,
            shuffle=True,
            mode="train",
            val_split=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
        val_generator = HDF5DataGenerator(
            file_paths=file_paths,
            feature_keys=feature_keys,
            batch_size=BATCH_SIZE,
            shuffle=True,
            mode="val",
            val_split=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        # Build model
        dnn_model = build_dnn_model(train_generator.input_dim, lr=1e-3)

        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            start_from_epoch=100,
            restore_best_weights=False,
        )

        # Train model
        print("Training model...")
        history = dnn_model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=EPOCHS,
            callbacks=[early_stop],
        )

        # Save model and history
        output_directory = os.path.join(output_path, "ML", dataset_str)
        ensure_dir_exists(output_directory)

        history_path = os.path.join(output_directory, f"{model_str}_history.h5")
        with h5py.File(history_path, "w") as f:
            for key, values in history.history.items():
                f.create_dataset(key, data=values)

        model_path = os.path.join(output_directory, f"{model_str}.h5")
        dnn_model.save(model_path)
        print(f"Model saved to {model_path}.")

        # Plot training history
        plot_training_history(history)

    if args.test:
        # Load model
        model_path = os.path.join(output_path, "ML", dataset_str, f"{model_str}.h5")
        model = keras.models.load_model(model_path, compile=True)

        if model is None:
            print("No model found.")
            return

        # Create test generator
        test_generator = HDF5DataGenerator(
            file_paths=file_paths,
            feature_keys=feature_keys,
            batch_size=BATCH_SIZE,
            shuffle=False,
            mode="test",
            val_split=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        # Evaluate model
        print("Evaluating model on test set...")
        loss, accuracy, auc = model.evaluate(test_generator, verbose=1)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test AUC: {auc:.4f}")

        # Predict and save predictions
        print("Generating predictions...")
        y_pred = model.predict(test_generator, verbose=1)

        # Load true labels manually
        y_true = np.concatenate([y for _, y in test_generator], axis=0)

        output_dir = os.path.join(output_path, "ML")
        ensure_dir_exists(output_dir)
        predictions_path = os.path.join(output_dir, "predictions.h5")

        with h5py.File(predictions_path, "w") as f:
            f.create_dataset("y_true", data=y_true)
            f.create_dataset("y_pred", data=y_pred)

        print(f"Predictions saved to {predictions_path}")

    if args.plot:
        print("Loading training history...")
        output_directory = os.path.join(output_path, "ML", dataset_str)
        history_path = os.path.join(output_directory, f"{model_str}_history.h5")

        with h5py.File(history_path, "r") as f:
            history = {key: list(f[key][:]) for key in f.keys()}

        plot_training_history(type("History", (), {"history": history}))


if __name__ == "__main__":
    main()
