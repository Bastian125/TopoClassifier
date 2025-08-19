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
data_files = {
    "20": {
        "a": "mc20a_withPU_raw.h5",
        "d": "mc20d_withPU_raw.h5",
        "e": "mc20e_withPU_raw.h5",
    },
    "23": {
        "a": "mc23a_withPU_raw.h5",
        "d": "mc23d_withPU_raw.h5",
        "e": "mc23e_withPU_raw.h5",
    },
}
data_files_noPU = {
    "20": {
        "a": "mc20a_noPU_raw.h5",
        "d": "mc20d_noPU_raw.h5",
        "e": "mc20e_noPU_raw.h5",
    },
    "23": {
        "a": "mc23a_noPU_raw.h5",
        "d": "mc23d_noPU_raw.h5",
        "e": "mc23e_noPU_raw.h5",
    },
}


# ---------- Argument Parser ---------- #
parser = argparse.ArgumentParser(
    description="Plot cluster features for MC20a/d/e and MC23a/d/e."
)
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
    help="Plot mean and median cluster response in n_PV bins for clusters with the complete energy range,"
    "clusters with energy lower than 100~GeV, and clusters with energy greater than or equal to 100~GeV ",
)
mode_group.add_argument("--all", action="store_true", help="Make every plot.")
args = parser.parse_args()


# ---------- Plot Config ---------- #
plt.rcParams.update(
    {
        "axes.prop_cycle": plt.cycler(
            color=[
                "#0072B2",  # Blue
                "#D55E00",  # Vermilion
                "#009E73",  # Bluish green
                "#E69F00",  # Orange
                "#56B4E9",  # Sky blue
                "#F0E442",  # Yellow
                "#CC79A7",  # Reddish purple
                "#000000",  # Black
            ]
        ),
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "legend.frameon": False,
        "legend.fontsize": 11,
        "font.size": 12,
        "text.usetex": True,
    }
)


# ---------- Helper Functions ---------- #
def load_feature(feature, campaign, subcampaign, PU=True):
    data = (
        data_files_noPU[str(campaign)][subcampaign]
        if not PU
        else data_files[str(campaign)][subcampaign]
    )
    file_path = os.path.join(data_save_path, data)
    print(f"Load {feature} for MC{campaign}{subcampaign} from {file_path}...")

    with h5py.File(file_path, "r") as f:
        return f[feature][:]


def plot_feature(
    feature,
    campaign,
    subcampaign,
    nbins,
    start,
    stop,
    logx=False,
    logy=False,
    xlabel=None,
    ylabel="Normalised",
    density=True,
    integer_bins=False,
):
    print(f"Plot {feature} for MC{campaign}{subcampaign}...")
    if xlabel is None:
        xlabel = feature

    data = load_feature(feature=feature, campaign=campaign, subcampaign=subcampaign)
    data = data[np.isfinite(data)]  # guard against NaN/Inf

    if integer_bins:
        mn, mx = int(np.floor(np.min(data))), int(np.ceil(np.max(data)))
        bins = np.arange(mn, mx + 1, 1)
        plt.xlim([mn - 0.5, mx + 0.5])
    elif logx:
        lo = max(start, np.nextafter(0, 1.0))  # avoid log10(0)
        bins = np.logspace(np.log10(lo), np.log10(stop), nbins)
        plt.xscale("log")
    else:
        bins = nbins
        plt.xlim([start, stop])

    n, _, _ = plt.hist(data, density=density, bins=bins, histtype="step")

    if logy and np.any(n > 0):
        plt.yscale("log")
        min_pos = np.min(n[n > 0])

        # pad one decade below the smallest positive bin height (cap at a tiny floor)
        plt.ylim(bottom=max(min_pos * 0.1, 1e-8))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    save_plot(f"{campaign}{subcampaign}/features", f"{feature}")
    plt.close()


def plot_response(campaign, subcampaign):
    """
    Plots response for one MC campaign and for different n_PV bins.
    Produces:
    - Full range (0–100)
    - Zoomed range (0–5)
    - Fine range (0–1)
    """
    response = load_feature("cluster_response", campaign, subcampaign=subcampaign)
    n_PV = load_feature("nPrimVtx", campaign, subcampaign=subcampaign)

    def plot_range(beginning, end, suffix, log_y=True):
        nbins = 100
        hrange = [beginning, end]
        lim = (beginning, end)

        plt.hist(
            response[(n_PV <= 10)],
            bins=nbins,
            range=hrange,
            histtype="step",
            density=True,
            label=r"$1 < n_{\mathrm{PV}} \leq 10$",
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

        if log_y:
            plt.yscale("log")

        plt.xlabel(r"Response")
        plt.ylabel("Normalised")
        plt.xlim(lim)
        plt.legend()
        plt.tight_layout()

        save_plot(f"{campaign}{subcampaign}/response", f"response_{suffix}")
        plt.close()

    # Create all three plots
    plot_range(0, 100, "full")
    plot_range(0.1, 5, "zoomed")
    plot_range(0.1, 1, "fine", log_y=False)  # Disable log scale for tiny range


def plot_response_with_and_with_out_PU(campaign, subcampaign):
    """
    Plots response for one MC campaign and for different n_PV bins.
    Includes:
    - Full range (0–100)
    - Zoomed range (0–5)
    - Fine range (0–1)
    """
    response = load_feature("cluster_response", campaign, subcampaign=subcampaign)
    response_noPU = load_feature(
        "cluster_response", campaign, subcampaign=subcampaign, PU=False
    )
    n_PV = load_feature("nPrimVtx", campaign, subcampaign=subcampaign)

    def plot_range(beginning, end, suffix, log_y=True):
        nbins = 100
        hrange = [beginning, end]
        lim = (beginning, end)

        plt.hist(
            response[(n_PV <= 10)],
            bins=nbins,
            range=hrange,
            histtype="step",
            density=True,
            label=r"$1 < n_{\mathrm{PV}} \leq 10$",
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

        if log_y:
            plt.yscale("log")
        plt.xlabel(r"Response")
        plt.ylabel("Normalised")
        plt.xlim(lim)
        plt.legend()
        plt.tight_layout()
        save_plot(
            f"{campaign}{subcampaign}/response",
            f"noPU_vs_PU_response_{campaign}_{suffix}",
        )
        plt.close()

    # Generate all three plots
    plot_range(0, 100, "full")
    plot_range(0.1, 5, "zoomed")
    plot_range(0.1, 1, "fine", log_y=False)  # log scale off for small values


def plot_mean_median_response(campaign, subcampaign, energy):
    """
    Plots mean and median response in n_PV bins between 10 and 50 for cluster with the complete energy range, clusters with energy less than 100~GeV, and clusters with energy greater than or equal to 100~GeV
    """
    if energy == "all":
        response = load_feature("cluster_response", campaign, subcampaign=subcampaign)
        clusterE = load_feature("clusterE", campaign, subcampaign=subcampaign)
        n_PV = load_feature("nPrimVtx", campaign, subcampaign=subcampaign)
    elif energy == "<100~GeV":
        response = load_feature("cluster_response", campaign, subcampaign=subcampaign)
        clusterE = load_feature("clusterE", campaign, subcampaign=subcampaign)
        n_PV = load_feature("nPrimVtx", campaign, subcampaign=subcampaign)

        # Apply cuts
        response = response[clusterE < 100]
        n_PV = n_PV[clusterE < 100]

    elif energy == ">=100~GeV":
        response = load_feature("cluster_response", campaign, subcampaign=subcampaign)
        clusterE = load_feature("clusterE", campaign, subcampaign=subcampaign)
        n_PV = load_feature("nPrimVtx", campaign, subcampaign=subcampaign)

        # Apply cuts
        response = response[clusterE >= 100]
        n_PV = n_PV[clusterE >= 100]

    n_PV_bins = np.arange(10, 50, 5)

    mean_response = []
    median_response = []
    mean_uncertainty = []
    median_uncertainty = []
    n_PV_centers = []

    for i in range(len(n_PV_bins) - 1):
        n_PV_min, n_PV_max = n_PV_bins[i], n_PV_bins[i + 1]
        n_PV_mask = (n_PV >= n_PV_min) & (n_PV < n_PV_max)
        responses_in_bin = response[n_PV_mask]
        N = len(responses_in_bin)

        if N > 1:
            sigma = np.std(responses_in_bin)
            mean_val = np.mean(responses_in_bin)
            median_val = np.median(responses_in_bin)
            sem = sigma / np.sqrt(N)
            sigma_median = 1.253 * sigma / np.sqrt(N)
        else:
            mean_val = np.nan
            median_val = np.nan
            sem = np.nan
            sigma_median = np.nan

        mean_response.append(mean_val)
        median_response.append(median_val)
        mean_uncertainty.append(sem)
        median_uncertainty.append(sigma_median)
        n_PV_centers.append((n_PV_min + n_PV_max) / 2)

    # Energy label for legend
    if energy == "all":
        energy_label = "All energies"
    elif energy == "<100~GeV":
        energy_label = r"$E_{\mathrm{clus}} < 100$ GeV"
    elif energy == ">=100~GeV":
        energy_label = r"$E_{\mathrm{clus}} \geq 100$ GeV"

    # Dummy line for energy category
    plt.plot([], [], " ", label=energy_label)
    plt.errorbar(
        n_PV_centers, mean_response, yerr=mean_uncertainty, fmt="o", label="Mean"
    )
    plt.errorbar(
        n_PV_centers, median_response, yerr=median_uncertainty, fmt="s", label="Median"
    )
    plt.xlim(10, 45)
    plt.xlabel(r"$N_{\mathrm{PV}}$")
    plt.ylabel(r"Response")
    plt.legend(loc="upper left")
    plt.tight_layout()
    if energy == "all":
        save_plot(f"{campaign}{subcampaign}/response", f"mean_median")
    elif energy == "<100~GeV":
        save_plot(f"{campaign}{subcampaign}/response", f"mean_median_<100~GeV")
    elif energy == ">=100~GeV":
        save_plot(f"{campaign}{subcampaign}/response", f"mean_median_>=100~GeV")
    plt.close()


def plot_run_comparison(features):
    """
    Plot each feature for concatenated Run 2 and Run 3 datasets.
    """
    subcampaigns = ["a", "d", "e"]

    for feature_key in features:
        settings = plot_settings[feature_key]
        feature = settings["feature"]
        nbins = settings["nbins"]
        start = settings["start"]
        stop = settings["stop"]
        log = settings.get("log", False)
        xlabel = settings.get("xlabel", feature)
        ylabel = settings.get("ylabel", "Normalised")
        density = settings.get("density", True)

        print(
            f"Comparing feature '{feature}' between MC20 and MC23 (all subcampaigns)..."
        )

        # Load and concatenate features across subcampaigns
        feature_20 = np.concatenate(
            [load_feature(feature, 20, sub) for sub in subcampaigns]
        )
        feature_23 = np.concatenate(
            [load_feature(feature, 23, sub) for sub in subcampaigns]
        )

        if log:
            bins = np.logspace(np.log10(start), np.log10(stop), nbins)
            plt.xscale("log")
        else:
            bins = nbins
            plt.xlim([start, stop])

        plt.hist(
            feature_20,
            density=density,
            bins=bins,
            histtype="step",
            label=r"Run 2",
        )
        plt.hist(
            feature_23,
            density=density,
            bins=bins,
            histtype="step",
            label=r"Run 3",
        )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        save_plot("run_comparison/MC20_vs_MC23", f"{feature}_run_comparison")
        plt.close()


def plot_features_overlayed_by_nPV_bins(subcampaign):
    """
    Plot all features for MC20 and MC23 campaigns with overlaid n_PV bins in one plot.
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

    for campaign in [20, 23]:
        n_PV = load_feature("nPrimVtx", campaign, subcampaign=subcampaign)

        for feature_key, settings in plot_settings.items():
            feature_name = settings["feature"]
            feature_data = load_feature(feature_name, campaign, subcampaign=subcampaign)
            nbins = settings["nbins"]
            start = settings["start"]
            stop = settings["stop"]
            log = settings.get("log", False)
            xlabel = settings.get("xlabel", feature_name)
            ylabel = settings.get("ylabel", "Normalised")
            density = settings.get("density", True)

            if log:
                bins = np.logspace(np.log10(start), np.log10(stop), nbins)
                plt.xscale("log")
            else:
                bins = nbins
                plt.xlim([start, stop])

            for (low, high), label in zip(nPV_bins, nPV_labels):
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
                )

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()

            save_plot(
                save_dir=f"{campaign}{subcampaign}/nPV_comparison",
                output_name=feature_name,
            )
            plt.close()


def plot_high_response(subcampaign):
    """
    Plot every feature for MC20 and MC23 campaigns, overlaying clusters with response <= 40 and > 40.
    """
    response_threshold = 40
    response_label = r"$R_{\mathrm{clus}}^{\mathrm{EM}}$"
    categories = [(None, response_threshold), (response_threshold, None)]
    category_labels = [
        rf"{response_label} $\leq$ {response_threshold}",
        rf"{response_label} $>$ {response_threshold}",
    ]

    for campaign in [20, 23]:
        cluster_response = load_feature(
            "cluster_response", campaign, subcampaign=subcampaign
        )

        for feature_key, settings in plot_settings.items():
            feature_name = settings["feature"]
            feature_data = load_feature(feature_name, campaign, subcampaign=subcampaign)
            nbins = settings["nbins"]
            start = settings["start"]
            stop = settings["stop"]
            log = settings.get("log", False)
            xlabel = settings.get("xlabel", feature_name)
            ylabel = settings.get("ylabel", "Normalised")
            density = settings.get("density", True)

            if log:
                bins = np.logspace(np.log10(start), np.log10(stop), nbins)
                plt.xscale("log")
            else:
                bins = nbins
                plt.xlim([start, stop])

            # Overlay for response <= 40 and > 40
            for (low, high), label in zip(categories, category_labels):
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
                )

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()

            save_plot(
                save_dir=f"{campaign}{subcampaign}/response_comparison",
                output_name=f"{feature_name}_high_response",
            )
            plt.close()


def plot_all_features(subcampaign):
    """
    Plot every feature in plot_settings for both MC20 and MC23 campaigns.
    """
    for campaign in [20, 23]:
        for feature_key, settings in plot_settings.items():
            feature = settings["feature"]
            nbins = settings["nbins"]
            start = settings["start"]
            stop = settings["stop"]
            integer_bins = settings["integer_bins"]
            logx = settings.get("logx", False)
            logy = settings.get("logy", False)
            xlabel = settings.get("xlabel", feature)
            ylabel = settings.get("ylabel", "Normalised")
            density = settings.get("density", True)

            print(f"Plotting {feature} for MC{campaign}{subcampaign}...")

            plot_feature(
                feature=feature,
                campaign=campaign,
                subcampaign=subcampaign,
                nbins=nbins,
                start=start,
                stop=stop,
                integer_bins=integer_bins,
                logx=logx,
                logy=logy,
                xlabel=xlabel,
                ylabel=ylabel,
                density=density,
            )


def save_plot(save_dir, output_name):
    """
    Saves plot to given save directory and output name.
    """
    save_path = os.path.join(output_path, save_dir)
    ensure_dir_exists(save_path)
    plt.savefig(os.path.join(save_path, output_name) + ".pdf")


# ---------- Main Function ---------- #
def main():
    subcampaigns = ["a", "d", "e"]

    for sub in subcampaigns:
        if args.all:
            plot_all_features(sub)

        if args.avgMu or args.all:
            feature = "avgMu"
            for campaign in [20, 23]:
                plot_feature(
                    feature=feature,
                    campaign=campaign,
                    subcampaign=sub,
                    nbins=40,
                    start=0,
                    stop=100,
                    xlabel=r"$\langle \mu \rangle$",
                    ylabel="Number of topoclusters",
                    integer_bins=True,
                )
                save_plot(
                    save_dir=f"{campaign}{sub}/summary",
                    output_name=feature,
                )

        if args.NPV or args.all:
            feature = "nPrimVtx"
            for campaign in [20, 23]:
                plot_feature(
                    feature=feature,
                    campaign=campaign,
                    subcampaign=sub,
                    nbins=50,
                    start=0,
                    stop=50,
                    xlabel=r"$n_{\mathrm{PV}}$",
                    ylabel="Number of topoclusters",
                    integer_bins=True,
                )
                save_plot(
                    save_dir=f"{campaign}{sub}/summary",
                    output_name=feature,
                )

        if args.run_comparison or args.all:
            plot_run_comparison(plot_settings)

        if args.NPV_comparison or args.all:
            plot_features_overlayed_by_nPV_bins(sub)

        if args.high_response or args.all:
            plot_high_response(sub)

        if args.response or args.all:
            for campaign in [20, 23]:
                plot_response(campaign, sub)

        if args.response_noPU_vs_PU or args.all:
            for campaign in [20, 23]:
                plot_response_with_and_with_out_PU(campaign, sub)

        if args.PU_response or args.all:
            for campaign in [20, 23]:
                for energy in ["all", "<100~GeV", ">=100~GeV"]:
                    plot_mean_median_response(campaign, sub, energy)


if __name__ == "__main__":
    main()
