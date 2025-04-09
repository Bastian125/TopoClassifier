"""Plots datasets."""

# ---------- Imports ---------- #
import os
import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt

from config import data_save_path, output_path
from io_utils import ensure_dir_exists

# ---------- Config ---------- #
data20 = "mc20e_withPU_raw.h5"
data23 = "mc23e_withPU_raw.h5"


# ---------- Argument Parser ---------- #
parser = argparse.ArgumentParser(description="Plot cluster features for MC20e/MC23e.")
mode_group = parser.add_mutually_exclusive_group(required=True)
mode_group.add_argument("--avgMu", action="store_true", help="Plot avgMu distribution.")
mode_group.add_argument("--NPV", action="store_true", help="Plot n_PV distribution.")
mode_group.add_argument(
    "--clusterE", action="store_true", help="Plot clusterE distribution."
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


def plot_feature(feature, campaign, nbins, start, stop, log=False):
    """Plot a single feature, linear or log."""
    print(f"Plot {feature} for MC{campaign}e...")
    feature_data = load_feature(feature=feature, campaign=campaign)

    if log:
        bins = np.logspace(np.log10(start), np.log10(stop), nbins)
        plt.xscale("log")
    else:
        bins = nbins
        plt.xlim([start, stop])

    plt.hist(feature_data, density=True, bins=bins, histtype="step")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.tight_layout()


# ---------- Main Function ---------- #
def main():
    ensure_dir_exists(output_path)

    if args.avgMu:
        feature = "avgMu"
        plot_feature(feature=feature, campaign=20, nbins=100, start=0, stop=100)
        save_path = os.path.join(output_path, "20")
        ensure_dir_exists(save_path)
        plt.savefig(os.path.join(save_path, f"{feature}_20.pdf"))
        plt.close()

    if args.clusterE:
        feature = "clusterE"
        plot_feature(
            feature=feature, campaign=20, nbins=50, start=1e-1, stop=1e2, log=True
        )
        save_path = os.path.join(output_path, "20")
        ensure_dir_exists(save_path)
        plt.savefig(os.path.join(save_path, f"{feature}_20.pdf"))
        plt.close()


if __name__ == "__main__":
    main()
