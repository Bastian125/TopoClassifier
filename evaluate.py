"""
Plots probabilites that cluster belongs to a given class, ROC-curve, and precision-recall curve.
"""

# ---------- Imports ---------- #
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


# ---------- Helper Functions ---------- #
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
    model, dataset, feature_names, prefix_path, device="cpu", batch_size=10000
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
