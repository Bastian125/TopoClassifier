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
    if args.avgMu:
        feature = "avgMu"
        plot_feature(
            feature=feature,
            campaign=20,
            nbins=40,
            start=0,
            stop=100,
            xlabel=r"$\langle \mu \rangle$",
            ylabel="Number of topoclusters",
        )
        save_plot(save_dir="20", output_name=f"{feature}_20")

    if args.NPV:
        feature = "NPV"
        plot_feature(
            feature="nPrimVtx",
            campaign=20,
            nbins=50,
            start=0,
            stop=50,
            xlabel=r"$n_{\mathrm{PV}}$",
            ylabel="Number of topoclusters",
        )
        save_plot(save_dir="20", output_name=f"{feature}_20")

    if args.clusterE:
        feature = "clusterE"
        plot_feature(
            feature=feature, campaign=20, nbins=50, start=1e-1, stop=1e2, log=True
        )
        save_plot(save_dir="20", output_name=f"{feature}_20")


if __name__ == "__main__":
    main()
