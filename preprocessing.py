"""
Preprocess root files for training or plotting and store them as hdf5-files.
Optimized for memory efficiency with chunking and compression.
Normalization is applied across full campaigns (e.g., mc20) using streaming statistics.
"""

# ---------- Imports ---------- #
import os
import argparse
import gc

import uproot
import h5py
import numpy as np
import pandas as pd

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
parser = argparse.ArgumentParser(description="Perform preprocessing of root files.")
mode_group = parser.add_mutually_exclusive_group(required=True)
mode_group.add_argument(
    "--campaign",
    type=str,
    choices=["mc20a", "mc20d", "mc20e", "mc23a", "mc23d", "mc23e", "mc20", "mc23"],
    help="Specify the campaign used for preprocessing.",
)
mode_group.add_argument(
    "--full", action="store_true", help="Run full preprocessing on all datasets"
)
parser.add_argument(
    "--no-normalisation",
    action="store_true",
    help="Skip normalisation and time transformation",
)
args = parser.parse_args()


# ---------- Helper Functions ---------- #
def apply_cuts(df):
    df = df[
        (df["cluster_ENG_CALIB_TOT"] > 0.3)
        & (df["clusterE"] > 0)
        & (df["cluster_CENTER_LAMBDA"] > 0.0)
        & (df["cluster_FIRST_ENG_DENS"] > 0.0)
        & (df["cluster_SECOND_TIME"] > 0.0)
        & (df["cluster_SIGNIFICANCE"] > 0.0)
    ].drop("cluster_SIGNIFICANCE", axis=1)
    return df


def apply_high_pile_up_cut(df):
    return df[(df["avgMu"] > 20)]


def compute_response(df):
    df["cluster_response"] = df["clusterE"] / df["cluster_ENG_CALIB_TOT"]
    return df[df["cluster_response"] > 0.1]


def load_and_process(file_path, label, apply_norm):
    print(f"Loading {file_path}...")
    root_file = uproot.open(file_path)
    tree = root_file["ClusterTree;1"]
    df = tree.arrays(columns, library="pd")
    length_before_cuts = len(df)
    df = apply_cuts(df)
    if label == 0 and apply_norm:
        df = apply_high_pile_up_cut(df)
    df["label"] = label
    df = compute_response(df)
    print(f"  -> {len(df)}/{length_before_cuts} entries retained after all cuts\n")
    return df


def split_data_full(df):
    trainval_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    train_df, val_df = train_test_split(
        trainval_df, test_size=0.25, stratify=trainval_df["label"], random_state=42
    )
    return train_df, val_df, test_df


def compute_streaming_stats(sub_tags):
    print("Pass 1: Streaming training stats...")
    sums = {}
    sqsums = {}
    mins = {}
    count = 0
    for feature in log_features + normal_features + ["cluster_time"]:
        sums[feature] = 0.0
        sqsums[feature] = 0.0
        mins[feature] = np.inf

    for sub_tag in sub_tags:
        df_withpu = load_and_process(
            os.path.join(root_path, f"{sub_tag}_withPU.root"), 0, apply_norm=False
        )
        df_nopu = load_and_process(
            os.path.join(root_path, f"{sub_tag}_noPU.root"), 1, apply_norm=False
        )
        df_combined = pd.concat([df_withpu, df_nopu], ignore_index=True)
        train_df, _, _ = split_data_full(df_combined)
        for feature in sums:
            vals = train_df[feature].values
            if feature == "cluster_time":
                vals = np.cbrt(np.abs(vals)) * np.sign(vals)
            sums[feature] += vals.sum()
            sqsums[feature] += (vals**2).sum()
            mins[feature] = min(mins[feature], vals.min())
        count += len(train_df)
        del df_withpu, df_nopu, df_combined, train_df
        gc.collect()

    stats = {}
    for feature in sums:
        mean = sums[feature] / count
        std = np.sqrt(sqsums[feature] / count - mean**2)
        shift = abs(mins[feature]) + 1e-6 if feature in log_features else 0
        stats[feature] = (mean, std, shift)

    return stats


def normalize_with_stats(train_df, val_df, test_df, stats, tag):
    stats_lines = [f"Normalization statistics for {tag}\n"]
    for feature in log_features:
        _, _, shift = stats[feature]
        for df in [train_df, val_df, test_df]:
            df[feature] = np.log10(df[feature] + shift)
        stats_lines.append(f"{feature} (log): shift = {shift:.6e}\n")

    for feature in normal_features:
        mean, std, _ = stats[feature]
        for df in [train_df, val_df, test_df]:
            df[feature] = (df[feature] - mean) / std
        stats_lines.append(f"{feature}: mean = {mean:.6f}, std = {std:.6f}\n")

    mean, std, _ = stats["cluster_time"]
    for df in [train_df, val_df, test_df]:
        x = np.cbrt(np.abs(df["cluster_time"])) * np.sign(df["cluster_time"])
        df["cluster_time"] = (x - mean) / std
    stats_lines.append(f"cluster_time (cbrt): mean = {mean:.6f}, std = {std:.6f}\n")

    stats_path = os.path.join(save_path, f"{tag}_norm_stats.txt")
    with open(stats_path, "w") as f:
        f.writelines(stats_lines)
    print(f"Saved normalization stats to {stats_path}")


def save_split(df, tag, split_name):
    output_path = os.path.join(save_path, f"{tag}_{split_name}.h5")
    with h5py.File(output_path, "w") as f:
        for col in df.columns:
            f.create_dataset(col, data=df[col].values, chunks=True, compression="gzip")
    print(f"Saved {split_name} split to {output_path}\n")


def merge_h5_files(file_list, output_path):
    with h5py.File(output_path, "w") as fout:
        dsets = {}
        for path in file_list:
            with h5py.File(path, "r") as fin:
                for key in fin:
                    if key not in dsets:
                        dsets[key] = fout.create_dataset(
                            key,
                            data=fin[key],
                            maxshape=(None,),
                            chunks=True,
                            compression="gzip",
                        )
                    else:
                        dsets[key].resize((dsets[key].shape[0] + fin[key].shape[0],))
                        dsets[key][-fin[key].shape[0] :] = fin[key][...]


def main():
    apply_norm = not args.no_normalisation

    if args.campaign:
        print("Single campaign mode activated...")
        tag = args.campaign

        if tag == "mc20":
            sub_tags = ["mc20a", "mc20d", "mc20e"]
        elif tag == "mc23":
            sub_tags = ["mc23a", "mc23d", "mc23e"]
        else:
            sub_tags = [tag]

        # Step 1: Compute streaming stats across all campaigns
        stats = compute_streaming_stats(sub_tags) if apply_norm else None
        suffix = "norm" if apply_norm else "raw"

        # Step 2: Process and save each campaign individually
        print("Pass 2: Normalizing and saving splits...")
        for sub_tag in sub_tags:
            df_withpu = load_and_process(
                os.path.join(root_path, f"{sub_tag}_withPU.root"), 0, apply_norm=False
            )
            df_nopu = load_and_process(
                os.path.join(root_path, f"{sub_tag}_noPU.root"), 1, apply_norm=False
            )
            df_combined = pd.concat([df_withpu, df_nopu], ignore_index=True)
            train_df, val_df, test_df = split_data_full(df_combined)
            if apply_norm:
                normalize_with_stats(train_df, val_df, test_df, stats, sub_tag)
            save_split(train_df, sub_tag, f"{suffix}_train")
            save_split(val_df, sub_tag, f"{suffix}_val")
            save_split(test_df, sub_tag, f"{suffix}_test")
            del df_withpu, df_nopu, df_combined, train_df, val_df, test_df
            gc.collect()

    elif args.full:
        print("Full mode activated...")
        all_tags = ["mc20a", "mc20d", "mc20e", "mc23a", "mc23d", "mc23e"]

        # Step 1: Compute streaming stats across all campaigns
        stats = compute_streaming_stats(all_tags) if apply_norm else None
        suffix = "norm" if apply_norm else "raw"

        # Step 2: Process and save each campaign individually
        for tag in all_tags:
            print(f"Processing {tag}...")

            df_withpu = load_and_process(
                os.path.join(root_path, f"{tag}_withPU.root"), 0, apply_norm=False
            )
            df_nopu = load_and_process(
                os.path.join(root_path, f"{tag}_noPU.root"), 1, apply_norm=False
            )
            df_combined = pd.concat([df_withpu, df_nopu], ignore_index=True)
            train_df, val_df, test_df = split_data_full(df_combined)
            if apply_norm:
                normalize_with_stats(train_df, val_df, test_df, stats, tag)
            save_split(train_df, tag, f"{suffix}_train")
            save_split(val_df, tag, f"{suffix}_val")
            save_split(test_df, tag, f"{suffix}_test")
            del df_withpu, df_nopu, df_combined, train_df, val_df, test_df
            gc.collect()


if __name__ == "__main__":
    main()
