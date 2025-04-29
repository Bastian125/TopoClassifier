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
from sklearn.metrics import roc_curve, auc

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
    "--mc20", action="store_true", help="Train model with MC20 data."
)
data_group.add_argument(
    "--mc23", action="store_true", help="Train model with MC23 data."
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
mode_group.add_argument("--train", action="store_true", help="Train ML model.")
mode_group.add_argument("--test", action="store_true", help="Test ML model.")
mode_group.add_argument("--plot", action="store_true", help="Plot training history.")
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
def build_dnn_model(input_dim, lr):
    print("Building DNN model...")
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
    print("Plotting training history...")
    fig, axs = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}
    )

    axs[0].plot(history["loss"], label="Training Loss")
    axs[0].plot(history["val_loss"], label="Validation Loss")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    loss_diff = np.array(history["val_loss"]) - np.array(history["loss"])
    axs[1].plot(loss_diff, label="val_loss - loss", linestyle="--")
    axs[1].axhline(0, color="black", linewidth=0.5)
    axs[1].set_ylabel("Loss Diff")
    axs[1].set_xlabel("Epoch")
    axs[1].grid(True)

    output_dir = os.path.join(output_path, "ML", dataset_str)
    ensure_dir_exists(output_dir)
    save_path = os.path.join(output_dir, f"{model_str}_training_history.pdf")
    plt.savefig(save_path)
    plt.close()
    print(f"Training history saved to {save_path}")


# ---------- Main Function ---------- #
def main():
    output_dir = os.path.join(output_path, "ML", dataset_str)
    ensure_dir_exists(output_dir)

    if args.train:
        print("Starting training...")

        if args.mc20:
            campaigns = ["mc20a", "mc20d", "mc20e"]
        elif args.mc23:
            campaigns = ["mc23a", "mc23d", "mc23e"]
        else:
            campaigns = None  # Special case for mc20a small run

        model = None
        full_history = {}

        if campaigns:
            for campaign in campaigns:
                print(f"Training on {campaign}...")
                file_paths = [
                    os.path.join(data_save_path, f"{campaign}_withPU_norm.h5"),
                    os.path.join(data_save_path, f"{campaign}_noPU_norm.h5"),
                ]

                train_gen = HDF5DataGenerator(
                    file_paths=file_paths,
                    feature_keys=feature_keys,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    mode="train",
                    val_split=TEST_SIZE,
                    random_state=RANDOM_STATE,
                )
                val_gen = HDF5DataGenerator(
                    file_paths=file_paths,
                    feature_keys=feature_keys,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    mode="val",
                    val_split=TEST_SIZE,
                    random_state=RANDOM_STATE,
                )

                if model is None:
                    model = build_dnn_model(train_gen.input_dim, lr=1e-3)

                history = model.fit(
                    train_gen,
                    validation_data=val_gen,
                    epochs=EPOCHS,
                    verbose=1,
                )

                for key in history.history.keys():
                    if key not in full_history:
                        full_history[key] = []
                    full_history[key].extend(history.history[key])

        else:
            # Only for mc20a (small test)
            file_paths = [
                os.path.join(data_save_path, "mc20a_withPU_norm.h5"),
                os.path.join(data_save_path, "mc20a_noPU_norm.h5"),
            ]

            train_gen = HDF5DataGenerator(
                file_paths,
                feature_keys,
                BATCH_SIZE,
                shuffle=True,
                mode="train",
                val_split=TEST_SIZE,
            )
            val_gen = HDF5DataGenerator(
                file_paths,
                feature_keys,
                BATCH_SIZE,
                shuffle=True,
                mode="val",
                val_split=TEST_SIZE,
            )

            model = build_dnn_model(train_gen.input_dim, lr=1e-3)

            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=EPOCHS,
                verbose=1,
            )

            full_history = history.history

        # Save final model and history
        model.save(os.path.join(output_dir, f"{model_str}.h5"))
        print(f"Model saved at {output_dir}")

        history_path = os.path.join(output_dir, f"{model_str}_history.h5")
        with h5py.File(history_path, "w") as f:
            for key, values in full_history.items():
                f.create_dataset(key, data=values)
        print(f"Training history saved at {history_path}")

        plot_training_history(full_history)

    if args.test:
        print("Starting model evaluation...")
        model_path = os.path.join(output_dir, f"{model_str}.h5")
        model = keras.models.load_model(model_path)

        # Create test generator
        if args.mc20a:
            test_files = [
                os.path.join(data_save_path, "mc20a_withPU_norm.h5"),
                os.path.join(data_save_path, "mc20a_noPU_norm.h5"),
            ]
        elif args.mc20:
            test_files = [
                os.path.join(data_save_path, "mc20a_withPU_norm.h5"),
                os.path.join(data_save_path, "mc20a_noPU_norm.h5"),
                os.path.join(data_save_path, "mc20d_withPU_norm.h5"),
                os.path.join(data_save_path, "mc20d_noPU_norm.h5"),
                os.path.join(data_save_path, "mc20e_withPU_norm.h5"),
                os.path.join(data_save_path, "mc20e_noPU_norm.h5"),
            ]
        elif args.mc23:
            test_files = [
                os.path.join(data_save_path, "mc23a_withPU_norm.h5"),
                os.path.join(data_save_path, "mc23a_noPU_norm.h5"),
                os.path.join(data_save_path, "mc23d_withPU_norm.h5"),
                os.path.join(data_save_path, "mc23d_noPU_norm.h5"),
                os.path.join(data_save_path, "mc23e_withPU_norm.h5"),
                os.path.join(data_save_path, "mc23e_noPU_norm.h5"),
            ]

        test_gen = HDF5DataGenerator(
            file_paths=test_files,
            feature_keys=feature_keys,
            batch_size=BATCH_SIZE,
            shuffle=False,
            mode="test",
            val_split=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        loss, accuracy, auc_score = model.evaluate(test_gen, verbose=1)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test AUC: {auc_score:.4f}")

        y_pred = model.predict(test_gen, verbose=1)
        y_true = test_gen.labels[test_gen.indices]

        # Save predictions
        with h5py.File(os.path.join(output_dir, "predictions.h5"), "w") as f:
            f.create_dataset("y_true", data=y_true)
            f.create_dataset("y_pred", data=y_pred)

        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{model_str}_roc_curve.pdf"))
        plt.close()
        print(f"ROC curve saved.")

    if args.plot:
        print("Plotting training history...")
        history_path = os.path.join(output_dir, f"{model_str}_history.h5")
        with h5py.File(history_path, "r") as f:
            history = {key: list(f[key][:]) for key in f.keys()}

        plot_training_history(history)


if __name__ == "__main__":
    main()
