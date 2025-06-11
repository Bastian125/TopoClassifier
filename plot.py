"""
Plots datasets.
"""

# ---------- Imports ---------- #
import os
import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt

from config import data_save_path, output_path, plot_settings
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
    "--avgMu",
    action="store_true",
    help="Plots distribution of avgMu for both campaigns",
)
mode_group.add_argument(
    "--NPV",
    action="store_true",
    help="Plots distribution of n_PV for both campaigns",
)
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
    "--high_response",
    action="store_true",
    help="Plot every feature of both campaigns for cluster response lower or equal to 40 or higher than 40 for comparison.",
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
mode_group.add_argument(
    "--PU_response",
    action="store_true",
    help="Plot mean and median cluster response in n_PV bins for clusters with the complete energy ramge,"
    "clusters with energy lower than 100~GeV, and clusters with energy greater than or equal to 100~GeV ",
)
args = parser.parse_args()


# ---------- Helper Functions ---------- #
def load_feature(feature, campaign, PU=True):
    """
    Load feature for MC20e or MC23e from HDF5 file.
    """
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
    """
    Plot a single feature, linear or log.
    """
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
    """
    Plots response for one MC campaign and for different n_PV bins.
    """
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
    plt.ylabel(r"Relative number of clusters")
    plt.xlim(lim)
    plt.legend()
    plt.tight_layout()
    save_plot("response", f"response_{campaign}")
    plt.close()


def plot_response_with_and_with_out_PU(campaign):
    """
    Plots response for one MC campaign and for different n_PV bins.
    """
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


def plot_mean_meadian_response(campaign, energy):
    """
    Plots mean and median response in n_PV bins between 10 and 50 for cluster with the complete energy range, clusters with energy less than 100~GeV, and clusters with energy greater than or equal to 100~GeV
    """
    if energy == "all":
        response = load_feature("cluster_response", campaign)
        clusterE = load_feature("clusterE", campaign)
        n_PV = load_feature("nPrimVtx", campaign)
    elif energy == "<100~GeV":
        response = load_feature("cluster_response", campaign)
        clusterE = load_feature("clusterE", campaign)
        n_PV = load_feature("nPrimVtx", campaign)

        # Apply cuts
        response = response[clusterE < 100]
        n_PV = n_PV[clusterE < 100]

    elif energy == ">=100~GeV":
        response = load_feature("cluster_response", campaign)
        clusterE = load_feature("clusterE", campaign)
        n_PV = load_feature("nPrimVtx", campaign)

        # Apply cuts
        response = response[clusterE >= 100]
        n_PV = n_PV[clusterE >= 100]

    n_PV_bins = np.arange(10, 50, 5)
    mean_response = []
    median_response = []
    n_PV_centers = []

    for i in range(len(n_PV_bins) - 1):
        n_PV_min, n_PV_max = n_PV_bins[i], n_PV_bins[i + 1]
        n_PV_mask = (n_PV >= n_PV_min) & (n_PV < n_PV_max)
        responses_in_bin = response[n_PV_mask]

        # Avoid empty bins
        if len(responses_in_bin) > 0:
            mean_val = np.mean(responses_in_bin)
            median_val = np.median(responses_in_bin)
        else:
            mean_val = np.nan
            median_val = np.nan

        # Store results
        mean_response.append(mean_val)
        median_response.append(median_val)
        n_PV_centers.append((n_PV_min + n_PV_max) / 2)

    plt.plot(n_PV_centers, mean_response, marker="o", linestyle="None", label="Mean")
    plt.plot(
        n_PV_centers, median_response, marker="o", linestyle="None", label="Median"
    )
    plt.xlim(10, 45)
    plt.xlabel(r"$N_{\mathrm{PV}}$")
    plt.ylabel(r"Response")
    plt.legend()
    plt.tight_layout()
    if energy == "all":
        save_plot("response", f"mean_median_{campaign}")
    elif energy == "<100~GeV":
        save_plot("response", f"mean_median_<100~GeV_{campaign}")
    elif energy == ">=100~GeV":
        save_plot("response", f"mean_median_>=100~GeV_{campaign}")
    plt.close()


def plot_run_comparison(features):
    """
    Plot each feature for Run 2 (MC20e) and Run 3 (MC23e) in one plot.
    """
    for feature_key in features:
        settings = plot_settings[feature_key]
        feature = settings["feature"]
        nbins = settings["nbins"]
        start = settings["start"]
        stop = settings["stop"]
        log = settings.get("log", False)
        xlabel = settings.get("xlabel", feature)
        ylabel = settings.get("ylabel", "Relative number of clusters")
        density = settings.get("density", True)

        print(f"Comparing feature '{feature}' between MC20e and MC23e...")

        feature_20 = load_feature(feature, 20)
        feature_23 = load_feature(feature, 23)

        if log:
            bins = np.logspace(np.log10(start), np.log10(stop), nbins)
            plt.xscale("log")
        else:
            bins = nbins
            plt.xlim([start, stop])

        plt.hist(feature_20, density=density, bins=bins, histtype="step", label="MC20e")
        plt.hist(feature_23, density=density, bins=bins, histtype="step", label="MC23e")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        save_plot("run_comparison", f"{feature}_run_comparison")
        plt.close()


def plot_features_overlayed_by_nPV_bins():
    """
    Plot all features for MC20e and MC23e campaigns with overlaid n_PV bins in one plot.
    """
    nPV_bins = [
        (None, 10),  # nPV < 10
        (10, 20),  # 10 <= nPV < 20
        (20, 30),  # 20 <= nPV < 30
        (30, None),  # nPV > 30
    ]
    nPV_labels = [
        r"$n_{\mathrm{PV}} < 10$",
        r"$10 \leq n_{\mathrm{PV}} < 20$",
        r"$20 \leq n_{\mathrm{PV}} < 30$",
        r"$n_{\mathrm{PV}} > 30$",
    ]
    colors = ["blue", "orange", "green", "red"]

    for campaign in [20, 23]:
        n_PV = load_feature("nPrimVtx", campaign)

        for feature_key, settings in plot_settings.items():
            feature_name = settings["feature"]
            feature_data = load_feature(feature_name, campaign)
            nbins = settings["nbins"]
            start = settings["start"]
            stop = settings["stop"]
            log = settings.get("log", False)
            xlabel = settings.get("xlabel", feature_name)
            ylabel = settings.get("ylabel", "Relative number of clusters")
            density = settings.get("density", True)

            if log:
                bins = np.logspace(np.log10(start), np.log10(stop), nbins)
                plt.xscale("log")
            else:
                bins = nbins
                plt.xlim([start, stop])

            # Create overlayed histograms
            for (low, high), label, color in zip(nPV_bins, nPV_labels, colors):
                if low is None:
                    mask = n_PV < high
                elif high is None:
                    mask = n_PV > low
                else:
                    mask = (n_PV >= low) & (n_PV < high)

                filtered_data = feature_data[mask]

                plt.hist(
                    filtered_data,
                    bins=bins,
                    density=density,
                    histtype="step",
                    label=label,
                    color=color,
                )

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()

            save_plot(
                save_dir=f"{campaign}", output_name=f"{feature_name}_nPV_comparison"
            )
            plt.close()


def plot_high_response():
    """
    Plot every feature for MC20e and MC23e campaigns, overlaying clusters with response <= 40 and > 40.
    """
    response_threshold = 40
    response_label = r"$R_{\mathrm{clus}}^{\mathrm{EM}}$"
    categories = [(None, response_threshold), (response_threshold, None)]
    category_labels = [
        rf"{response_label} $\leq$ {response_threshold}",
        rf"{response_label} $>$ {response_threshold}",
    ]
    colors = ["blue", "red"]

    for campaign in [20, 23]:
        cluster_response = load_feature("cluster_response", campaign)

        for feature_key, settings in plot_settings.items():
            feature_name = settings["feature"]
            feature_data = load_feature(feature_name, campaign)
            nbins = settings["nbins"]
            start = settings["start"]
            stop = settings["stop"]
            log = settings.get("log", False)
            xlabel = settings.get("xlabel", feature_name)
            ylabel = settings.get("ylabel", "Relative number of clusters")
            density = settings.get("density", True)

            if log:
                bins = np.logspace(np.log10(start), np.log10(stop), nbins)
                plt.xscale("log")
            else:
                bins = nbins
                plt.xlim([start, stop])

            # Overlay for response <= 40 and > 40
            for (low, high), label, color in zip(categories, category_labels, colors):
                if low is None:
                    mask = cluster_response <= high
                elif high is None:
                    mask = cluster_response > low
                else:
                    mask = (cluster_response > low) & (cluster_response <= high)

                filtered_data = feature_data[mask]

                plt.hist(
                    filtered_data,
                    bins=bins,
                    density=density,
                    histtype="step",
                    label=label,
                    color=color,
                )

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()

            save_plot(
                save_dir=f"{campaign}/response_comparison",
                output_name=f"{feature_name}_high_response",
            )
            plt.close()


def save_plot(save_dir, output_name):
    """
    Saves plot to given save directory and output name.
    """
    save_path = os.path.join(output_path, save_dir)
    ensure_dir_exists(save_path)
    plt.savefig(os.path.join(save_path, output_name) + ".pdf")
    plt.close()


# ---------- Main Function ---------- #
def main():
    if args.avgMu:
        feature = "avgMu"
        for campaign in [20, 23]:
            plot_feature(
                feature=feature,
                campaign=campaign,
                nbins=40,
                start=0,
                stop=100,
                xlabel=r"$\langle \mu \rangle$",
                ylabel="Number of topoclusters",
            )
            save_plot(save_dir=f"{campaign}", output_name=f"{feature}_{campaign}")

    if args.NPV:
        feature = "nPrimVtx"
        for campaign in [20, 23]:
            plot_feature(
                feature=feature,
                campaign=campaign,
                nbins=50,
                start=0,
                stop=50,
                xlabel=r"$n_{\mathrm{PV}}$",
                ylabel="Number of topoclusters",
            )
            save_plot(save_dir=f"{campaign}", output_name=f"{feature}_{campaign}")

    if args.run_comparison:
        plot_run_comparison(plot_settings)

    if args.NPV_comparison:
        plot_features_overlayed_by_nPV_bins()

    if args.high_response:
        plot_high_response()

    if args.response:
        for campaign in [20, 23]:
            plot_response(campaign)

    if args.response_noPU_vs_PU:
        for campaign in [20, 23]:
            plot_response_with_and_with_out_PU(campaign)

    if args.PU_response:
        for campaign in [20, 23]:
            for energy in ["all", "<100~GeV", ">=100~GeV"]:
                plot_mean_meadian_response(campaign, energy)


if __name__ == "__main__":
    main()
