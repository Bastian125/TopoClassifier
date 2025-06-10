"""
Preprocesses withPU and noPU .root files to hdf5 files with cuts applied.
"""

import os
import argparse
import numpy as np
import h5py
import uproot

from sklearn.model_selection import train_test_split

from config import (
    columns,
    log_features,
    normal_features,
    data_root_path as root_path,
    data_save_path as save_path,
)

from io_utils import ensure_dir_exists

# ---------- Argument Parser ---------- #
parser = argparse.ArgumentParser(description="Preprocess ROOT files and HDF5 splits.")
mode_group = parser.add_mutually_exclusive_group(required=True)
mode_group.add_argument(
    "--campaign",
    type=str,
    choices=["mc20a", "mc20d", "mc20e", "mc23a", "mc23d", "mc23e", "mc20", "mc23"],
    help="Specify the campaign for preprocessing or renormalisation.",
)
mode_group.add_argument(
    "--full", action="store_true", help="Run full preprocessing on all datasets."
)
mode_group.add_argument(
    "--print_features", action="store_true", help="Print all features in the root file."
)
parser.add_argument(
    "--no-normalisation",
    action="store_true",
    help="Skip normalisation and time transformation.",
)
args = parser.parse_args()


# ---------- Helper Functions ---------- #
def list_root_features(file_path):
    """
    List all available branches (features) in the ROOT file under 'ClusterTree;1'.

    Parameters:
        file_path (str): Path to the ROOT file.

    Prints:
        List of feature names found in the ClusterTree.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with uproot.open(file_path) as f:
        if "ClusterTree;1" not in f:
            print(f"'ClusterTree;1' not found in file: {file_path}")
            return

        tree = f["ClusterTree;1"]
        feature_names = tree.keys()
        print(f"Features in {file_path}:\n")
        for name in feature_names:
            print(name)


def apply_cuts_mask(df):
    """
    Returns boolean mask for physics-motivated cuts on clusters.
    """
    return (
        (df["cluster_ENG_CALIB_TOT"] > 0.3)
        & (df["clusterE"] > 0)
        & (df["cluster_CENTER_LAMBDA"] > 0.0)
        & (df["cluster_FIRST_ENG_DENS"] > 0.0)
        & (df["cluster_SECOND_TIME"] > 0.0)
        & (df["cluster_SIGNIFICANCE"] > 0.0)
    )


def apply_high_pile_up_cut_mask(df):
    """
    Returns mask for avgMu > 20 to select pile-up dominated events (used only for background).
    """
    return df["avgMu"] > 20


def compute_response_and_mask(df):
    """
    Computes response as clusterE / cluster_ENG_CALIB_TOT, adds 'cluster_response', and mask for response > 0.1.
    """
    response = df["clusterE"] / df["cluster_ENG_CALIB_TOT"]
    df["cluster_response"] = response
    return response > 0.1


def load_and_process(file_path, label, apply_norm):
    """
    Load ROOT file, apply physics cuts, pile-up cut (if label 0), and compute response.
    """
    print(f"Loading {file_path}...")
    tree = uproot.open(file_path)["ClusterTree;1"]
    df = tree.arrays(columns, library="np")
    length_before_cuts = len(df["clusterE"])
    mask = apply_cuts_mask(df)
    if label == 0 and apply_norm:
        mask &= apply_high_pile_up_cut_mask(df)
    mask &= compute_response_and_mask(df)
    df = {key: val[mask] for key, val in df.items()}
    df["label"] = np.full_like(df["clusterE"], label)
    print(
        f"  -> {len(df['clusterE'])}/{length_before_cuts} entries retained after all cuts\n"
    )
    return df


def split_data_full(df):
    """
    Stratified split into 60% train, 20% val, 20% test. Returns dicts of numpy arrays.
    """
    labels = df["label"]
    indices = np.arange(len(labels))
    trainval_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=0.25, stratify=labels[trainval_idx], random_state=42
    )
    return (
        {key: val[train_idx] for key, val in df.items()},
        {key: val[val_idx] for key, val in df.items()},
        {key: val[test_idx] for key, val in df.items()},
    )


def normalize_data(train, val, test, tag):
    """
    Recompute mean/std for all features. Log10 features are already log-transformed, so mean/std only.
    """
    stats_lines = [f"Normalization statistics for {tag}\n"]
    for feature in log_features + normal_features:
        finite_mask = np.isfinite(train[feature])
        if not np.any(finite_mask):
            print(f"Warning: No finite values for {feature}. Skipping normalization.")
            continue
        mean = train[feature][finite_mask].mean()
        std = train[feature][finite_mask].std()
        for split in [train, val, test]:
            split[feature] = (split[feature] - mean) / std
            split[feature][~np.isfinite(split[feature])] = 0
        stats_lines.append(f"{feature}: mean = {mean:.6f}, std = {std:.6f}\n")
    x_train_time = np.cbrt(np.abs(train["cluster_time"])) * np.sign(
        train["cluster_time"]
    )
    finite_mask = np.isfinite(x_train_time)
    mean = x_train_time[finite_mask].mean()
    std = x_train_time[finite_mask].std()
    for split in [train, val, test]:
        x = np.cbrt(np.abs(split["cluster_time"])) * np.sign(split["cluster_time"])
        split["cluster_time"] = (x - mean) / std
        split["cluster_time"][~np.isfinite(split["cluster_time"])] = 0
    stats_lines.append(f"cluster_time (cbrt): mean = {mean:.6f}, std = {std:.6f}\n")
    stats_path = os.path.join(save_path, f"{tag}_stats.txt")
    with open(stats_path, "w") as f:
        f.writelines(stats_lines)
    print(f"Saved normalization stats to {stats_path}")


def save_split(df, base_name, tag):
    """
    Save a dictionary of numpy arrays as an HDF5 file.
    """
    output_path = os.path.join(save_path, f"{base_name}_{tag}.h5")
    with h5py.File(output_path, "w") as f:
        for key, val in df.items():
            f.create_dataset(key, data=val)
    print(f"Saved {tag} split to {output_path}")


def concatenate_and_renormalise(campaign, sub_campaigns):
    """
    Concatenate train/val/test splits of sub-campaigns, recompute mean/std, reapply normalization.
    """
    required_files = [
        os.path.join(save_path, f"{sub}_norm_train.h5") for sub in sub_campaigns
    ]
    if not all(os.path.exists(f) for f in required_files):
        print(f"Skipping renormalisation: missing required files for {campaign}.")
        return
    print(f"\nRenormalising and concatenating {campaign}...")
    combined_data = {split: {} for split in ["train", "val", "test"]}
    for split in ["train", "val", "test"]:
        for sub in sub_campaigns:
            file_path = os.path.join(save_path, f"{sub}_norm_{split}.h5")
            if not os.path.exists(file_path):
                print(f"Missing {file_path}, skipping renormalisation.")
                return
            with h5py.File(file_path, "r") as f:
                for key in f.keys():
                    combined_data[split].setdefault(key, []).append(f[key][:])
        combined_data[split] = {
            key: np.concatenate(arrays) for key, arrays in combined_data[split].items()
        }
    train = combined_data["train"]
    stats_lines = [f"Recomputed normalization statistics for {campaign}\n"]
    norm_stats = {}
    for feature in log_features + normal_features:
        finite_mask = np.isfinite(train[feature])
        if not np.any(finite_mask):
            print(f"Warning: No finite values for {feature}. Skipping normalization.")
            continue
        mean = train[feature][finite_mask].mean()
        std = train[feature][finite_mask].std()
        norm_stats[feature] = (mean, std)
        stats_lines.append(f"{feature}: mean = {mean:.6f}, std = {std:.6f}\n")
    x = np.cbrt(np.abs(train["cluster_time"])) * np.sign(train["cluster_time"])
    finite_mask = np.isfinite(x)
    mean_time = x[finite_mask].mean()
    std_time = x[finite_mask].std()
    stats_lines.append(
        f"cluster_time (cbrt): mean = {mean_time:.6f}, std = {std_time:.6f}\n"
    )
    stats_path = os.path.join(save_path, f"{campaign}_norm_stats.txt")
    with open(stats_path, "w") as f:
        f.writelines(stats_lines)
    print(f"Saved new normalization stats to {stats_path}")
    for split, data in combined_data.items():
        for feature in log_features + normal_features:
            if feature not in norm_stats:
                continue
            mean, std = norm_stats[feature]
            data[feature] = (data[feature] - mean) / std
            data[feature][~np.isfinite(data[feature])] = 0
        x = np.cbrt(np.abs(data["cluster_time"])) * np.sign(data["cluster_time"])
        data["cluster_time"] = (x - mean_time) / std_time
        data["cluster_time"][~np.isfinite(data["cluster_time"])] = 0
        output_path = os.path.join(save_path, f"{campaign}_norm_{split}.h5")
        with h5py.File(output_path, "w") as f_out:
            for key, val in data.items():
                f_out.create_dataset(key, data=val)
        print(f"Saved renormalised concatenated {split} split to {output_path}")


# ---------- Main ---------- #
def main():
    if args.print_features:
        list_root_features(os.path.join(root_path, "mc20a_withPU.root"))


    apply_norm = not args.no_normalisation
    if args.campaign:
        tag = args.campaign
        if tag in ["mc20", "mc23"]:
            sub_tags = {
                "mc20": ["mc20a", "mc20d", "mc20e"],
                "mc23": ["mc23a", "mc23d", "mc23e"],
            }[tag]
            concatenate_and_renormalise(tag, sub_tags)
            return
        print("Preprocessing from ROOT files...")
        df_withpu = load_and_process(
            os.path.join(root_path, f"{tag}_withPU.root"),
            label=0,
            apply_norm=apply_norm,
        )
        df_nopu = load_and_process(
            os.path.join(root_path, f"{tag}_noPU.root"), label=1, apply_norm=apply_norm
        )
        combined = {
            key: np.concatenate([df_withpu[key], df_nopu[key]])
            for key in df_withpu.keys()
        }
        train, val, test = split_data_full(combined)
        suffix = "norm" if apply_norm else "raw"
        if apply_norm:
            normalize_data(train, val, test, tag)
        save_split(train, tag, f"{suffix}_train")
        save_split(val, tag, f"{suffix}_val")
        save_split(test, tag, f"{suffix}_test")
    elif args.full:
        print("Full mode activated...")
        tags = ["mc20a", "mc20d", "mc20e", "mc23a", "mc23d", "mc23e"]
        for tag in tags:
            print(f"Processing {tag}...")
            df_withpu = load_and_process(
                os.path.join(root_path, f"{tag}_withPU.root"),
                label=0,
                apply_norm=apply_norm,
            )
            df_nopu = load_and_process(
                os.path.join(root_path, f"{tag}_noPU.root"),
                label=1,
                apply_norm=apply_norm,
            )
            combined = {
                key: np.concatenate([df_withpu[key], df_nopu[key]])
                for key in df_withpu.keys()
            }
            train, val, test = split_data_full(combined)
            suffix = "norm" if apply_norm else "raw"
            if apply_norm:
                normalize_data(train, val, test, tag)
            save_split(train, tag, f"{suffix}_train")
            save_split(val, tag, f"{suffix}_val")
            save_split(test, tag, f"{suffix}_test")


if __name__ == "__main__":
    main()
