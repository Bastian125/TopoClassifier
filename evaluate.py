"""
Plots probabilites that cluster belongs to a given class, ROC-curve, and precision-recall curve.
"""

# ---------- Imports ---------- #
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


# ---------- Helper Functions ---------- #
def plot_prediction_probabilities(y_true, y_pred):
    """
    Plot predicted probabilites that a cluster belongs to a given class.
    """

    plt.hist(y_pred[y_true == 0], bins=50, alpha=0.5, label="Pile-up", density=True)
    plt.hist(
        y_pred[y_true == 1], bins=50, alpha=0.5, label="Hard Scatter", density=True
    )
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.title("Prediction Probabilities")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_pred):
    """
    Plots true positive rate against false positive rate.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(y_true, y_pred):
    """
    Plots precision against recall of the model.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)

    plt.plot(
        recall, precision, color="green", lw=2, label=f"PR curve (AP = {pr_auc:.2f})"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------- Main Function ---------- #
def main():
    # Load predictions
    with h5py.File("predictions.h5", "r") as f:
        y_true = f["y_true"][:]
        y_pred = f["y_pred"][:].flatten()

    # Plot predictions and evaluation curves
    plot_prediction_probabilities(y_true, y_pred)
    plot_roc_curve(y_true, y_pred)
    plot_precision_recall_curve(y_true, y_pred)


if __name__ == "__main__":
    main()
