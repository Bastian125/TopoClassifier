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
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm

from config import data_save_path, output_path, feature_keys, jet_feature_keys
from io_utils import ensure_dir_exists
from dataloader import HDF5Dataset, JetGraphIterableDataset
from models import DNNModel, GAT
from evaluate import (
    plot_roc_curve,
    plot_precision_recall,
    plot_prediction_histogram,
    plot_permutation_importance,
    plot_cluster_response_comparison_histogram,
    remove_prefix_from_state_dict,
)


# ---------- File Config ---------- #
LEARNING_RATE_RUN2 = 1e-3
LEARNING_RATE_RUN3 = 1e-5
BATCH_SIZE_RUN2 = 512
BATCH_SIZE_RUN3 = 1024
EPOCHS = 400
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- Argument Parser ---------- #
parser = argparse.ArgumentParser(
    description="Train and/or test ML models on specific ATLAS campaigns."
)

parser.add_argument(
    "--train_campaign",
    type=str,
    required=True,
    choices=["mc20a", "mc20d", "mc20e", "mc23a", "mc23d", "mc23e", "mc20", "mc23"],
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
parser.add_argument(
    "--feature_importance",
    action="store_true",
    help="Plots feature importance for model trained on --train_campaign and tested on --test_campaign.",
)

model_group = parser.add_mutually_exclusive_group(required=True)
model_group.add_argument(
    "--DNN",
    action="store_true",
    help="DNN model that classifies hard-scatter and pile-up clusters.",
)
model_group.add_argument(
    "--JetDNN",
    action="store_true",
    help="DNN model that classifies hard-scatter and pile-up clusters with cluster and jet features.",
)
model_group.add_argument(
    "--GAT",
    action="store_true",
    help="Graph Attention Network (GAT) for topo-cluster classification.",
)


args = parser.parse_args()


# --------- Define Dataset and Model Type --------- #
def campaign_to_dataset(campaign):
    """Map campaign name to output folder name."""
    mapping = {
        "mc20a": "Run2a",
        "mc20d": "Run2d",
        "mc20e": "Run2e",
        "mc23a": "Run3a",
        "mc23d": "Run3d",
        "mc23e": "Run3e",
        "mc20": "Run2",
        "mc23": "Run3",
    }
    if campaign not in mapping:
        raise ValueError("Unknown campaign: " + campaign)
    return mapping[campaign]


train_dataset_str = campaign_to_dataset(args.train_campaign)
if args.DNN:
    model_str = "DNN"
elif args.JetDNN:
    model_str = "JetDNN"
elif args.GAT:
    model_str = "GAT"


# Set learning rate and batch size depending on Run2 or Run3
if "Run2" in train_dataset_str:
    LEARNING_RATE = LEARNING_RATE_RUN2
    BATCH_SIZE = BATCH_SIZE_RUN2
else:
    LEARNING_RATE = LEARNING_RATE_RUN3
    BATCH_SIZE = BATCH_SIZE_RUN3


# ---------- Helper Functions ---------- #
def plot_training_history(history):
    """Plot and save training vs validation loss curves."""
    print("Plotting training history...")
    fig, axs = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}
    )

    # Top plot: Loss curves
    axs[0].plot(history["train_loss"], label="Training Loss")
    axs[0].plot(history["val_loss"], label="Validation Loss")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_xlim(0, len(history["train_loss"]))

    # Annotate min val loss

    min_epoch = np.argmin(history["val_loss"]) + 1
    min_val = history["val_loss"][min_epoch - 1]

    axs[0].annotate(
        f"Min Val Loss\nEpoch {min_epoch}",
        xy=(min_epoch, min_val),
        xycoords="data",
        xytext=(30, 150),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        fontsize=9,
        ha="center"
    )


    # Bottom plot: Loss difference
    loss_diff = np.array(history["val_loss"]) - np.array(history["train_loss"])
    axs[1].plot(loss_diff, label="val_loss - loss", linestyle="--")
    axs[1].axhline(0, color="black", linewidth=0.5)
    axs[1].set_ylabel("Loss Diff")
    axs[1].set_xlabel("Epoch")
    axs[1].grid(True)
    axs[1].set_xlim(0, len(history["train_loss"]))

    output_dir = os.path.join(output_path, "ML", train_dataset_str)
    ensure_dir_exists(output_dir)
    save_path = os.path.join(output_dir, f"{model_str}_training_history.pdf")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history saved to {save_path}")


def train_model(model, train_loader, val_loader, criterion, optimizer):
    """Train model with early stopping and mixed precision."""
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience, wait = 20, 0
    scaler = GradScaler()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for X, y in loop:
            X, y = X.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                outputs = model(X)
                loss = criterion(outputs, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE).unsqueeze(1)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(
                output_path, "ML", train_dataset_str, f"{model_str}_best.pt"
            )
            torch.save(model.state_dict(), best_model_path)
            wait = 0
        else:
            wait += 1
            if epoch >= 100 and wait >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load(best_model_path))
    return model, history


def train_GNN(train_dataset, val_dataset, input_dim, output_dir, model_str):
    """Train GAT model with early stopping and save results like the DNN block."""
    train_loader = GeoDataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4)
    val_loader = GeoDataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)
# ---------- Paths ---------- #
    model = GAT(input_dim).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()

    best_val_loss = float("inf")
    wait = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(EPOCHS):
        print(f"Starting epoch {epoch+1}/{EPOCHS}", flush=True)
        start_time = time.time()

        model.train()
        train_loss = 0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type=DEVICE.type):
                outputs = model(batch.x, batch.edge_index)
                loss = criterion(outputs.view(-1), batch.y.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= max(1, num_batches)

        # Validation loop
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                outputs = model(batch.x, batch.edge_index)
                loss = criterion(outputs.view(-1), batch.y.float())
                val_loss += loss.item()
                val_batches += 1
        val_loss /= max(1, val_batches)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1:3d} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Time: {elapsed:.1f}s"
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{model_str}_best.pt"),
            )
            wait = 0
        else:
            wait += 1
            if epoch >= 100 and wait >= 20:
                print("Early stopping triggered.")
                break

    # Final save
    model.load_state_dict(torch.load(os.path.join(output_dir, f"{model_str}_best.pt")))
    torch.save(model.state_dict(), os.path.join(output_dir, f"{model_str}.pt"))

    # Save history
    with h5py.File(os.path.join(output_dir, f"{model_str}_history.h5"), "w") as f:
        for k, v in history.items():
            f.create_dataset(k, data=v)

    plot_training_history(history)
    return model, history

# ---------- Paths ---------- #
def get_predictions(model, loader):
    """Run inference and return true and predicted probabilities."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            outputs = torch.sigmoid(model(X)).cpu().numpy().flatten()
            y_true.extend(y.numpy())
            y_pred.extend(outputs)
    return np.array(y_true), np.array(y_pred)


def get_predictions_GNN(model, loader):
    """Run inference and return true and predicted probabilities for GNN."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            outputs = (
                torch.sigmoid(model(batch.x, batch.edge_index)).cpu().numpy().flatten()
            )
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(outputs)
    return np.array(y_true), np.array(y_pred)


# ---------- Main ---------- #
def main():
    output_dir = os.path.join(output_path, "ML", train_dataset_str)
    ensure_dir_exists(output_dir)

    if not args.test_campaign and not args.plot:
        print("Start training on:", args.train_campaign)

        if model_str == "DNN":
            train_file = os.path.join(
                data_save_path, f"{args.train_campaign}_norm_train.h5"
            )
            val_file = os.path.join(
                data_save_path, f"{args.train_campaign}_norm_val.h5"
            )

            train_dataset = HDF5Dataset(train_file, feature_keys, "label")
            val_dataset = HDF5Dataset(val_file, feature_keys, "label")
            input_dim = train_dataset[0][0].shape[0]

            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
            )

            with h5py.File(train_file, "r") as f:
                pos_weight = f.attrs["pos_weight"]
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(DEVICE)

            uncompiled_model = DNNModel(input_dim).to(DEVICE)
            model = torch.compile(uncompiled_model)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

            model, history = train_model(
                model, train_loader, val_loader, criterion, optimizer
            )

            torch.save(
                uncompiled_model.state_dict(),
                os.path.join(output_dir, f"{model_str}.pt"),
            )

            with h5py.File(
                os.path.join(output_dir, f"{model_str}_history.h5"), "w"
            ) as f:
                for key, values in history.items():
                    f.create_dataset(key, data=values)

            plot_training_history(history)

        if model_str == "JetDNN":
            train_file = os.path.join(
                data_save_path, f"{args.train_campaign}_norm_train.h5"
            )
            val_file = os.path.join(
                data_save_path, f"{args.train_campaign}_norm_val.h5"
            )

            train_dataset = HDF5Dataset(train_file, jet_feature_keys, "label")
            val_dataset = HDF5Dataset(val_file, jet_feature_keys, "label")
            input_dim = train_dataset[0][0].shape[0]

            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
            )

            with h5py.File(train_file, "r") as f:
                pos_weight = f.attrs["pos_weight"]
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(DEVICE)

            uncompiled_model = DNNModel(input_dim).to(DEVICE)
            model = torch.compile(uncompiled_model)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

            model, history = train_model(
                model, train_loader, val_loader, criterion, optimizer
            )

            torch.save(
                uncompiled_model.state_dict(),
                os.path.join(output_dir, f"{model_str}.pt"),
            )

            with h5py.File(
                os.path.join(output_dir, f"{model_str}_history.h5"), "w"
            ) as f:
                for key, values in history.items():
                    f.create_dataset(key, data=values)

            plot_training_history(history)

        if model_str == "GAT":
            train_pattern = os.path.join(
                data_save_path, f"{args.train_campaign}_graph_train_chunk*.pt"
            )
            val_pattern = os.path.join(
                data_save_path, f"{args.train_campaign}_graph_val_chunk*.pt"
            )

            train_dataset = JetGraphIterableDataset(train_pattern)
            val_dataset = JetGraphIterableDataset(val_pattern)

            # Peek to determine input dimension
            first_graph_batch = next(iter(train_dataset))
            input_dim = first_graph_batch.x.shape[1]

            print("Initialise and train model...")
            model, history = train_GNN(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                input_dim=input_dim,
                output_dir=output_dir,
                model_str=model_str,
            )

            plot_training_history(history)

    if args.test_campaign:
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

        if model_str == "DNN":
            model_cls = DNNModel
            model = model_cls(len(feature_keys)).to(DEVICE)
        elif model_str == "JetDNN":
            model_cls = DNNModel
            model = model_cls(len(jet_feature_keys)).to(DEVICE)

        state_dict = torch.load(os.path.join(output_dir, f"{model_str}_best.pt"))
        state_dict = remove_prefix_from_state_dict(state_dict)
        model.load_state_dict(state_dict)

        test_file = os.path.join(data_save_path, f"{args.test_campaign}_norm_test.h5")
        if model_str == "DNN":
            test_dataset = HDF5Dataset(test_file, feature_keys, "label")
        elif model_str == "JetDNN":
            test_dataset = HDF5Dataset(test_file, jet_feature_keys, "label")
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        y_true, y_pred = get_predictions(model, test_loader)

        with h5py.File(
            os.path.join(
                test_out_dir, f"{model_str}_on_{args.test_campaign}_predictions.h5"
            ),
            "w",
        ) as f:
            f.create_dataset("y_true", data=y_true)
            f.create_dataset("y_pred", data=y_pred)

        roc_prefix_test = os.path.join(
            test_out_dir, f"{model_str}_on_{args.test_campaign}"
        )
        # Start plotting
        plot_roc_curve(y_true, y_pred, prefix_path=roc_prefix_test)
        plot_precision_recall(y_true, y_pred, prefix_path=roc_prefix_test)
        plot_prediction_histogram(y_true, y_pred, prefix_path=roc_prefix_test)
        if args.feature_importance:
            plot_permutation_importance(
                model=model,
                dataset=test_dataset,
                feature_names=feature_keys,
                prefix_path=roc_prefix_test,
            )

        # Load from threshold TXT
        with open(roc_prefix_test + "_threshold.txt") as f:
            for line in f:
                if line.startswith("Best threshold:"):
                    best_threshold = float(line.split(":")[1].strip())

        with h5py.File(test_file, "r") as f:
            cluster_response = f["cluster_response"][:]

        plot_cluster_response_comparison_histogram(
            true_response=cluster_response,
            y_pred_probs=y_pred,
            prefix_path=roc_prefix_test,
            threshold=best_threshold,
        )

    if args.plot:
        history_path = os.path.join(output_dir, f"{model_str}_history.h5")
        if not os.path.exists(history_path):
            print(f"Error: No training history found at {history_path}")
        else:
            print(f"Loading training history from {history_path}")
            with h5py.File(history_path, "r") as f:
                history = {k: f[k][()] for k in f}
            history = {k: v.tolist() for k, v in history.items()}
            plot_training_history(history)


if __name__ == "__main__":
    main()
