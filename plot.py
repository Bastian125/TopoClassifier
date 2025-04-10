"""Plots datasets."""

# ---------- Imports ---------- #
import os
import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt

from config import data_save_path, output_path
from io_utils import ensure_dir_exists

# ---------- File Config ---------- #
data20 = "mc20e_withPU_raw.h5"
data23 = "mc23e_withPU_raw.h5"


# ---------- Argument Parser ---------- #
parser = argparse.ArgumentParser(description="Plot cluster features for MC20e/MC23e.")
mode_group = parser.add_mutually_exclusive_group(required=True)
mode_group.add_argument(
    "--run_comparison",
    action="store_true",
    help="Plot comparison of every feature for Run 2 and Run 3.",
)
mode_group.add_argument(
    "--NPV_comparison", action="store_true", help="Plot every feature."
)
args = parser.parse_args()


# ---------- Helper Functions ---------- #
def load_feature(feature, campaign):
    """Load feature for MC20e or MC23e from HDF5 file."""
    if campaign == 20:
        data = data20
    elif campaign == 23:
        data = data23

    file_path = os.path.join(data_save_path, data)
    print(f"Load {feature} for MC{campaign}e from {file_path}...")

    with h5py.File(file_path, "r") as f:
        return f[feature][:]


def plot_feature(
    feature,
    campaign,
    nbins,
    start,
    stop,
    log=False,
    xlabel=None,
    ylabel="Relative number of clusters",
    density=True,
):
    """Plot a single feature, linear or log."""
    print(f"Plot {feature} for MC{campaign}e...")
    if xlabel is None:
        xlabel = feature
    feature_data = load_feature(feature=feature, campaign=campaign)

    if log:
        bins = np.logspace(np.log10(start), np.log10(stop), nbins)
        plt.xscale("log")
    else:
        bins = nbins
        plt.xlim([start, stop])

    plt.hist(feature_data, density=density, bins=bins, histtype="step")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()


def save_plot(save_dir, output_name):
    """Saves plot to given save directory and output name."""
    save_path = os.path.join(output_path, save_dir)
    ensure_dir_exists(save_path)
    plt.savefig(os.path.join(save_path, output_name) + ".pdf")
    plt.close()


# ---------- Main Function ---------- #
def main():
    return 0


if __name__ == "__main__":
    main()
