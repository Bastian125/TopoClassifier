"""
Train different ML models on given input features.
"""

# ---------- Imports ---------- #
import os
import argparse
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
EPOCHS = 50

# ---------- Argument Parser ---------- #
parser = argparse.ArgumentParser(
    description="Train and/or test ML models on specific ATLAS campaigns."
)

parser.add_argument(
    "--train_campaign",
    type=str,
    required=True,
    choices=["mc20a", "mc20d", "mc20e", "mc23a", "mc23d", "mc23e"],
    help="Specify the campaign used for training.",
)

parser.add_argument(
    "--test_campaign",
    type=str,
    choices=["mc20a", "mc20d", "mc20e", "mc23a", "mc23d", "mc23e"],
    help="Optionally test the model trained on --train_campaign against this campaign.",
)

parser.add_argument(
    "--plot",
    action="store_true",
    help="Plot training history of the model trained on --train_campaign.",
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

args = parser.parse_args()


# --------- Define Dataset and Model Type --------- #
def campaign_to_dataset(campaign):
    mapping = {
        "mc20a": "Run2a",
        "mc20d": "Run2d",
        "mc20e": "Run2e",
        "mc23a": "Run3a",
        "mc23d": "Run3d",
        "mc23e": "Run3e",
    }
    if campaign not in mapping:
        raise ValueError("Unknown campaign: " + campaign)
    return mapping[campaign]


train_dataset_str = campaign_to_dataset(args.train_campaign)

if args.DNN1:
    model_str = "DNN1"
elif args.DNN2:
    model_str = "DNN2"
else:
    raise ValueError("No model type selected")


# ---------- Helper Functions ---------- #
def build_dnn_model(input_dim, lr): ...


def plot_training_history(history): ...


def plot_roc_curve(y_true, y_pred, save_path): ...


# ---------- Main Function ---------- #
def main():
    output_dir = os.path.join(output_path, "ML", train_dataset_str)
    ensure_dir_exists(output_dir)

    if not args.test_campaign:
        print("Start training on:", args.train_campaign)

        file_paths = [
            os.path.join(data_save_path, f"{args.train_campaign}_withPU_norm.h5"),
            os.path.join(data_save_path, f"{args.train_campaign}_noPU_norm.h5"),
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

        model = build_dnn_model(train_gen.input_dim, lr=1e-3)

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            verbose=1,
            callbacks=[early_stop],
        )

        full_history = history.history

        model.save(os.path.join(output_dir, f"{model_str}.h5"))
        print(f"Model saved at {output_dir}")

        history_path = os.path.join(output_dir, f"{model_str}_history.h5")
        with h5py.File(history_path, "w") as f:
            for key, values in full_history.items():
                f.create_dataset(key, data=values)
        print(f"Training history saved at {history_path}")

        plot_training_history(full_history)

    if args.test_campaign:
        test_dataset_str = campaign_to_dataset(args.test_campaign)
        test_out_dir = os.path.join(
            output_path, "ML", train_dataset_str, f"test_on_{args.test_campaign}"
        )
        ensure_dir_exists(test_out_dir)

        print(
            "Evaluating model trained on",
            args.train_campaign,
            "against",
            args.test_campaign,
        )
        model_path = os.path.join(
            output_path, "ML", train_dataset_str, f"{model_str}.h5"
        )
        model = keras.models.load_model(model_path)

        test_files = [
            os.path.join(data_save_path, f"{args.test_campaign}_withPU_norm.h5"),
            os.path.join(data_save_path, f"{args.test_campaign}_noPU_norm.h5"),
        ]

        test_gen = HDF5DataGenerator(
            file_paths=test_files,
            feature_keys=feature_keys,
            batch_size=BATCH_SIZE,
            shuffle=False,
            mode="all",
            val_split=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        loss, accuracy, auc_score = model.evaluate(test_gen, verbose=1)
        print(
            f"Test Loss: {loss:.4f}\nTest Accuracy: {accuracy:.4f}\nTest AUC: {auc_score:.4f}"
        )

        y_pred = model.predict(test_gen, verbose=1)
        y_true = test_gen.labels[test_gen.indices]

        with h5py.File(
            os.path.join(
                test_out_dir, f"{model_str}_on_{args.test_campaign}_predictions.h5"
            ),
            "w",
        ) as f:
            f.create_dataset("y_true", data=y_true)
            f.create_dataset("y_pred", data=y_pred)

        roc_path = os.path.join(
            test_out_dir, f"{model_str}_on_{args.test_campaign}_roc_curve.pdf"
        )
        plot_roc_curve(y_true, y_pred, save_path=roc_path)

    if args.plot:
        print("Plotting training history...")
        history_path = os.path.join(
            output_path, "ML", train_dataset_str, f"{model_str}_history.h5"
        )
        with h5py.File(history_path, "r") as f:
            history = {key: list(f[key][:]) for key in f.keys()}
        plot_training_history(history)


if __name__ == "__main__":
    main()
