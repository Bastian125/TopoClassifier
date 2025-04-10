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
data_noPU_20 = "mc20e_noPU_raw.h5"
data_noPU_23 = "mc23e_noPU_raw.h5"


# ---------- Argument Parser ---------- #
parser = argparse.ArgumentParser(description="Plot cluster features for MC20e/MC23e.")
mode_group = parser.add_mutually_exclusive_group(required=True)
mode_group.add_argument(
    "--run_comparison",
    action="store_true",
    help="Plot comparison of every feature for Run 2 and Run 3.",
)
mode_group.add_argument(
    "--NPV_comparison",
    action="store_true",
    help="Plot every feature for different n_PV bins for both campaigns.",
)
mode_group.add_argument(
    "--response",
    action="store_true",
    help="Creates response plots for different n_PV bins for both campaigns.",
)
mode_group.add_argument(
    "--response_noPU_vs_PU",
    action="store_true",
    help="Creates response plots for different n_PV bins, and no pile-up for both campaigns.",
)
args = parser.parse_args()


# ---------- Helper Functions ---------- #
def load_feature(feature, campaign, PU=True):
    """Load feature for MC20e or MC23e from HDF5 file."""
    if PU == False:
        if campaign == 20:
            data = data_noPU_20
        elif campaign == 23:
            data = data_noPU_23
    elif PU == True:
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


def plot_response(campaign):
    """Plots response for one MC campaign and for different n_PV bins."""
    response = load_feature("cluster_response", campaign)
    n_PV = load_feature("nPrimVtx", campaign)

    nbins = 100
    beginning = 0
    end = 100
    hrange = [beginning, end]
    lim = (beginning, end)

    plt.hist(
        response[(n_PV <= 10)],
        bins=nbins,
        range=hrange,
        histtype="step",
        density=True,
        label=r"$ 1 < n_{\mathrm{PV}} \leq 10$",
    )
    plt.hist(
        response[(n_PV <= 20) & (n_PV > 10)],
        bins=nbins,
        range=hrange,
        histtype="step",
        density=True,
        label=r"$10 < n_{\mathrm{PV}} \leq 20$",
    )
    plt.hist(
        response[(n_PV <= 30) & (n_PV > 20)],
        bins=nbins,
        range=hrange,
        histtype="step",
        density=True,
        label=r"$20 < n_{\mathrm{PV}} \leq 30$",
    )
    plt.hist(
        response[(n_PV > 30)],
        bins=nbins,
        range=hrange,
        histtype="step",
        density=True,
        label=r"$n_{\mathrm{PV}} > 30$",
    )
    plt.yscale("log")
    plt.xlabel(r"Response")
    plt.ylabel(r"Number of clusters")
    plt.xlim(lim)
    plt.legend()
    plt.tight_layout()
    save_plot("response", f"response_{campaign}")
    plt.close()


def plot_response_with_and_with_out_PU(campaign):
    """Plots response for one MC campaign and for different n_PV bins."""
    response = load_feature("cluster_response", campaign)
    response_noPU = load_feature("cluster_response", campaign, PU=False)
    n_PV = load_feature("nPrimVtx", campaign)

    nbins = 100
    beginning = 0
    end = 100
    hrange = [beginning, end]
    lim = (beginning, end)

    plt.hist(
        response[(n_PV <= 10)],
        bins=nbins,
        range=hrange,
        histtype="step",
        density=True,
        label=r"$ 1 < n_{\mathrm{PV}} \leq 10$",
    )
    plt.hist(
        response[(n_PV <= 20) & (n_PV > 10)],
        bins=nbins,
        range=hrange,
        histtype="step",
        density=True,
        label=r"$10 < n_{\mathrm{PV}} \leq 20$",
    )
    plt.hist(
        response[(n_PV <= 30) & (n_PV > 20)],
        bins=nbins,
        range=hrange,
        histtype="step",
        density=True,
        label=r"$20 < n_{\mathrm{PV}} \leq 30$",
    )
    plt.hist(
        response[(n_PV > 30)],
        bins=nbins,
        range=hrange,
        histtype="step",
        density=True,
        label=r"$n_{\mathrm{PV}} > 30$",
    )
    plt.hist(
        response_noPU,
        bins=nbins,
        range=hrange,
        histtype="step",
        density=True,
        label="No pile-up",
    )
    plt.yscale("log")
    plt.xlabel(r"Response")
    plt.ylabel(r"Number of clusters")
    plt.xlim(lim)
    plt.legend()
    plt.tight_layout()
    save_plot("response", f"noPU_vs_PU_response_{campaign}")
    plt.close()


def save_plot(save_dir, output_name):
    """Saves plot to given save directory and output name."""
    save_path = os.path.join(output_path, save_dir)
    ensure_dir_exists(save_path)
    plt.savefig(os.path.join(save_path, output_name) + ".pdf")
    plt.close()


# ---------- Main Function ---------- #
def main():
    if args.response:
        for campaign in [20, 23]:
            plot_response(campaign)

    if args.response_noPU_vs_PU:
        for campaign in [20, 23]:
            plot_response_with_and_with_out_PU(campaign)


if __name__ == "__main__":
    main()
