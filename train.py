"""
Train different ML models on given input features.
Updated to use pre-split HDF5 files: *_train.h5, *_val.h5, *_test.h5.
Rewritten in PyTorch with tqdm progress bars, mixed precision training, and BCEWithLogitsLoss.
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
from torch.amp import GradScaler, autocast
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from config import data_save_path, output_path
from io_utils import ensure_dir_exists
from dataloader import HDF5Dataset

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
    "--TunedDNN",
    action="store_true",
    help="DNN model that classifies pile-up only and mixed clusters.",
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
model_str = "DNN"

# Set learning rate and batch size depending on Run2 or Run3
if "Run2" in train_dataset_str:
    LEARNING_RATE = LEARNING_RATE_RUN2
    BATCH_SIZE = BATCH_SIZE_RUN2
else:
    LEARNING_RATE = LEARNING_RATE_RUN3
    BATCH_SIZE = BATCH_SIZE_RUN3


# ---------- Model Definition ---------- #
class DNNModel(nn.Module):
    """
    Deep Neural Network model for binary classification.
    Final layer does NOT include sigmoid; use BCEWithLogitsLoss instead.
    """

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
        )

    def forward(self, x):
        return self.model(x)


# ---------- Helper Functions ---------- #
def remove_prefix_from_state_dict(state_dict, prefix="_orig_mod."):
    """Remove prefix from state_dict keys. Storing the compiled model in PyTorch
    adds _orig_mod which must be removed for proper testing."""
    return {
        k.replace(prefix, "") if k.startswith(prefix) else k: v
        for k, v in state_dict.items()
    }


def plot_training_history(history):
    """Plot and save training vs validation loss curves."""
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


def plot_roc_curve(y_true, y_pred, prefix_path):
    """
    Plot and save ROC curve with best threshold using Youden's J statistic.
    Saves both PDF and TXT file with threshold and metrics.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Best threshold by maximizing TPR - FPR (Youden's J statistic)
    j_scores = tpr - fpr
    j_best_idx = np.argmax(j_scores)
    best_threshold = thresholds[j_best_idx]
    best_fpr = fpr[j_best_idx]
    best_tpr = tpr[j_best_idx]

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.scatter(
        best_fpr,
        best_tpr,
        color="red",
        zorder=5,
        label=f"Best threshold = {best_threshold:.4f}",
    )
    plt.text(
        best_fpr + 0.02,
        best_tpr - 0.05,
        f"Thresh = {best_threshold:.4f}\nTPR = {best_tpr:.3f}\nFPR = {best_fpr:.3f}",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.8
        ),
    )
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.grid(True)

    # Save
    pdf_path = prefix_path + "_roc_curve.pdf"
    txt_path = prefix_path + "_threshold.txt"
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close()

    with open(txt_path, "w") as f:
        f.write(f"Best threshold: {best_threshold:.6f}\n")
        f.write(f"True Positive Rate (TPR): {best_tpr:.6f}\n")
        f.write(f"False Positive Rate (FPR): {best_fpr:.6f}\n")
        f.write(f"AUC: {roc_auc:.6f}\n")

    print(f"ROC curve saved to {pdf_path}")
    print(f"Threshold info saved to {txt_path}")


def plot_precision_recall(y_true, y_pred, prefix_path):
    """
    Plot and save Precision-Recall (PR) curve with average precision.
    Saves both PDF and TXT file with summary.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall, precision, label=f"Avg Precision = {avg_precision:.4f}", linewidth=2
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="upper right")
    plt.grid(True)

    pr_pdf_path = prefix_path + "_pr_curve.pdf"
    pr_txt_path = prefix_path + "_pr_metrics.txt"

    plt.tight_layout()
    plt.savefig(pr_pdf_path)
    plt.close()

    with open(pr_txt_path, "w") as f:
        f.write(f"Average Precision: {avg_precision:.6f}\n")

    print(f"Precision-Recall curve saved to {pr_pdf_path}")
    print(f"PR metrics saved to {pr_txt_path}")


def plot_prediction_histogram(y_true, y_pred, prefix_path):
    """
    Plot a histogram of predicted probabilities for each class (label 0 and 1).
    Saves as a PDF file.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    plt.figure(figsize=(8, 6))
    plt.hist(
        y_pred[y_true == 1],
        bins=50,
        alpha=0.6,
        label="Hard-scatter",
        density=True,
        histtype="stepfilled",
    )
    plt.hist(
        y_pred[y_true == 0],
        bins=50,
        alpha=0.6,
        label="Mixed/Pile-up",
        density=True,
        histtype="stepfilled",
    )
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)

    hist_path = prefix_path + "_prob_hist.pdf"
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()
    print(f"Probability histogram saved to {hist_path}")


def compute_iqr(x):
    """
    Computes interquartile range to be used in cluster_response comparisons.
    """
    q75, q25 = np.percentile(x, [75, 25])
    return q75 - q25


def plot_cluster_response_comparison_histogram(
    true_response, y_pred_probs, prefix_path, threshold=0.5
):
    """
    Plot step histogram of cluster_response:
    - Full distribution
    - Subset selected by model as hard-scatter (y_pred ≥ threshold)

    Args:
        true_response (np.ndarray): All cluster_response values.
        y_pred_probs (np.ndarray): Predicted probabilities (after sigmoid).
        prefix_path (str): Output prefix for saving.
        threshold (float): Classification threshold.
    """
    true_response = np.asarray(true_response)
    y_pred_probs = np.asarray(y_pred_probs)

    if len(true_response) != len(y_pred_probs):
        raise ValueError("Length mismatch between true_response and y_pred_probs.")

    # Mask for DNN-selected hard-scatter clusters
    selection_mask = y_pred_probs >= threshold
    selected_response = true_response[selection_mask]

    # Compute iqr
    iqr_full = compute_iqr(true_response)
    iqr_selected = compute_iqr(selected_response)

    # Skip if empty
    if selected_response.size == 0:
        print("No selected clusters above threshold. Skipping response plot.")
        return

    # Plotting
    plt.figure(figsize=(8, 6))
    nbins = 100
    beginning = 0
    end = 100
    hrange = [beginning, end]
    lim = (beginning, end)

    plt.hist(
        true_response,
        bins=nbins,
        range=hrange,
        histtype="step",
        density=True,
        label=f"All clusters IQR = {iqr_full:.2f}",
    )
    plt.hist(
        selected_response,
        bins=nbins,
        range=hrange,
        histtype="step",
        density=True,
        label=f"Selected (≥ {threshold:.2f})\nIQR = {iqr_selected:.2f}",
    )

    plt.yscale("log")
    plt.xlabel("Response")
    plt.ylabel("Relative number of clusters")
    plt.xlim(lim)
    plt.legend()
    plt.tight_layout()

    out_path = prefix_path + "_cluster_response_hist.pdf"
    plt.savefig(out_path)
    plt.close()
    print(f"Step histogram saved to {out_path}")


def plot_permutation_importance(
    model, dataset, feature_names, prefix_path, device="cpu", batch_size=1024
):
    """
    Computes and plots feature permutation importance using AUC drop.
    Uses DataLoader batching to prevent memory issues.
    """
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    original_X, y_true = dataset.get_all_data()
    y_true_tensor = torch.tensor(y_true, dtype=torch.float32)

    # Baseline predictions
    base_preds = []
    base_loader = DataLoader(
        TensorDataset(torch.tensor(original_X, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for batch in base_loader:
            inputs = batch[0].to(device)
            outputs = torch.sigmoid(model(inputs)).cpu().numpy().flatten()
            base_preds.extend(outputs)
    base_score = roc_auc_score(y_true, base_preds)

    # Permutation importance
    importances = []

    for i, feat in enumerate(feature_names):
        X_permuted = original_X.copy()
        np.random.shuffle(X_permuted[:, i])

        perm_preds = []
        perm_loader = DataLoader(
            TensorDataset(torch.tensor(X_permuted, dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=False,
        )

        with torch.no_grad():
            for batch in perm_loader:
                inputs = batch[0].to(device)
                outputs = torch.sigmoid(model(inputs)).cpu().numpy().flatten()
                perm_preds.extend(outputs)

        perm_score = roc_auc_score(y_true, perm_preds)
        importances.append(base_score - perm_score)

    # Sort and plot
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = np.array(feature_names)[sorted_idx]
    sorted_importances = np.array(importances)[sorted_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importances)
    plt.xlabel("Drop in AUC")
    plt.title("Permutation Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plot_path = prefix_path + "_permutation_importance.pdf"
    plt.savefig(plot_path)
    plt.close()
    print(f"Permutation importance plot saved to {plot_path}")


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

            y_train = train_dataset.labels.numpy()
            weights = compute_class_weight(
                class_weight="balanced", classes=np.unique(y_train), y=y_train
            )
            class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
            pos_weight = class_weights[1] / class_weights[0]

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

            # Get train ROC, threshold and training history
            y_true_train, y_pred_train = get_predictions(model, train_loader)
            roc_prefix_train = os.path.join(
                output_dir, f"{model_str}_on_{args.train_campaign}_train"
            )
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

        model_cls = DNNModel
        model = model_cls(len(feature_keys)).to(DEVICE)
        state_dict = torch.load(os.path.join(output_dir, f"{model_str}_best.pt"))
        state_dict = remove_prefix_from_state_dict(state_dict)
        model.load_state_dict(state_dict)

        test_file = os.path.join(data_save_path, f"{args.test_campaign}_norm_test.h5")
        test_dataset = HDF5Dataset(test_file, feature_keys, "label")
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
