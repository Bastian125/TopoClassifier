"""
Train different ML models on given input features.
Updated to use pre-split HDF5 files: *_train.h5, *_val.h5, *_test.h5.
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

BATCH_SIZE = 512
EPOCHS = 400

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

    output_dir = os.path.join(output_path, "ML", train_dataset_str)
    ensure_dir_exists(output_dir)
    save_path = os.path.join(output_dir, f"{model_str}_training_history.pdf")
    plt.savefig(save_path)
    plt.close()
    print(f"Training history saved to {save_path}")


def plot_roc_curve(y_true, y_pred, save_path):
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
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve saved to {save_path}")


# ---------- Main Function ---------- #
def main():
    output_dir = os.path.join(output_path, "ML", train_dataset_str)
    ensure_dir_exists(output_dir)

    if not args.test_campaign and not args.plot:
        print("Start training on:", args.train_campaign)

        train_file = os.path.join(
            data_save_path, f"{args.train_campaign}_norm_train.h5"
        )
        val_file = os.path.join(data_save_path, f"{args.train_campaign}_norm_val.h5")

        train_gen = HDF5DataGenerator(
            file_path=train_file,
            feature_keys=feature_keys,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        val_gen = HDF5DataGenerator(
            file_path=val_file,
            feature_keys=feature_keys,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        model = build_dnn_model(train_gen.input_dim, lr=1e-3)

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            start_from_epoch=100,
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

    if args.plot:
        # Load and plot training history from saved HDF5
        history_path = os.path.join(
            output_path, "ML", train_dataset_str, f"{model_str}_history.h5"
        )
        if not os.path.exists(history_path):
            print(f"Error: No training history found at {history_path}")
        else:
            print(f"Loading training history from {history_path}")
            history_data = {}
            with h5py.File(history_path, "r") as f:
                for key in f:
                    history_data[key] = f[key][()]
            # Convert bytes to str keys if needed (optional)
            history_data = {k: v.tolist() for k, v in history_data.items()}
            plot_training_history(history_data)

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

        model_path = os.path.join(output_dir, f"{model_str}.h5")
        model = keras.models.load_model(model_path)

        test_file = os.path.join(data_save_path, f"{args.test_campaign}_norm_test.h5")

        test_gen = HDF5DataGenerator(
            file_path=test_file,
            feature_keys=feature_keys,
            batch_size=BATCH_SIZE,
            shuffle=False,
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


if __name__ == "__main__":
    main()
