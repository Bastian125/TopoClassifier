"""
Train different ML models on given input features.
Updated to use pre-split HDF5 files: *_train.h5, *_val.h5, *_test.h5.
Rewritten in PyTorch.
"""

# ---------- Imports ---------- #
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from config import data_save_path, output_path
from io_utils import ensure_dir_exists
from dataloader_torch import HDF5Dataset

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

LEARNING_RATE = 1e-3
BATCH_SIZE = 512
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

model_str = "DNN1" if args.DNN1 else "DNN2"


# ---------- Model Definition ---------- #
class DNNModel(nn.Module):
    def __init__(self, input_dim):
        super(DNNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# ---------- Helper Functions ---------- #
def plot_training_history(history):
    print("Plotting training history...")
    fig, axs = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}
    )

    axs[0].plot(history["train_loss"], label="Training Loss")
    axs[0].plot(history["val_loss"], label="Validation Loss")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    loss_diff = np.array(history["val_loss"]) - np.array(history["train_loss"])
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
    fpr, tpr, _ = roc_curve(y_true, y_pred)
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


def train_model(model, train_loader, val_loader, criterion, optimizer):
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience, wait = 20, 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for X, y in loop:
            X, y = X.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
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
            torch.save(model.state_dict(), "best_model.pt")
            wait = 0
        else:
            wait += 1
            if epoch >= 100 and wait >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load("best_model.pt"))
    return model, history


def get_predictions(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            outputs = model(X).cpu().numpy().flatten()
            y_true.extend(y.numpy())
            y_pred.extend(outputs)
    return np.array(y_true), np.array(y_pred)


# ---------- Main ---------- #
def main():
    output_dir = os.path.join(output_path, "ML", train_dataset_str)
    ensure_dir_exists(output_dir)

    if not args.test_campaign and not args.plot:
        print("Start training on:", args.train_campaign)

        train_file = os.path.join(
            data_save_path, f"{args.train_campaign}_norm_train.h5"
        )
        val_file = os.path.join(data_save_path, f"{args.train_campaign}_norm_val.h5")

        train_dataset = HDF5Dataset(train_file, feature_keys)
        val_dataset = HDF5Dataset(val_file, feature_keys)
        input_dim = train_dataset[0][0].shape[0]

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        y_train = train_dataset.labels.numpy()
        weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(y_train), y=y_train
        )
        class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
        pos_weight = class_weights[1] / class_weights[0]

        model = DNNModel(input_dim).to(DEVICE)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer
        )

        torch.save(model.state_dict(), os.path.join(output_dir, f"{model_str}.pt"))

        with h5py.File(os.path.join(output_dir, f"{model_str}_history.h5"), "w") as f:
            for key, values in history.items():
                f.create_dataset(key, data=values)
        plot_training_history(history)

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

        model = DNNModel(len(feature_keys)).to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(output_dir, f"{model_str}.pt")))

        test_file = os.path.join(data_save_path, f"{args.test_campaign}_norm_test.h5")
        test_dataset = HDF5Dataset(test_file, feature_keys)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        loss_fn = nn.BCELoss()
        loss, accuracy, auc_score = 0.0, 0.0, 0.0  # Optional: implement test metrics

        y_true, y_pred = get_predictions(model, test_loader)

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
