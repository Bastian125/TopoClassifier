"""
Preprocess root files for training or plotting and store them as hdf5-files.
"""

# ---------- Imports ---------- #
import os
import argparse

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
    "--test", action="store_true", help="Run in test mode (process only mc20e)"
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
    """
    Apply consistent cuts to all datasets, including ENG_CALIB_TOT > 0.3.
    """
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
    """
    Apply avgMu cut so that the background data truly reflects pile-up dominated clusters.
    """
    return df[(df["avgMu"] > 20)]


def compute_response(df):
    """
    Compute cluster response (clusterE/cluster_ENG_CALIB_TOT).
    """
    df["cluster_response"] = df["clusterE"] / df["cluster_ENG_CALIB_TOT"]


def load_and_process(file_path, label, apply_norm):
    """
    Load a single ROOT file, apply cuts and label, and return a DataFrame.
    """
    print(f"Loading {file_path}...")
    root_file = uproot.open(file_path)
    tree = root_file["ClusterTree;1"]
    df = tree.arrays(columns, library="pd")
    df = apply_cuts(df)
    if label == 0 and apply_norm:
        df = apply_high_pile_up_cut(df)
    df["label"] = label
    compute_response(df)
    return df


def split_data(df):
    """
    Stratified split of dataframe into train, val, and test sets.
    """
    df_train, df_temp = train_test_split(
        df, test_size=0.4, stratify=df["label"], random_state=42
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=0.5, stratify=df_temp["label"], random_state=42
    )
    return df_train, df_val, df_test


def normalize_data(train_df, val_df, test_df):
    """
    Apply log and normalization transforms using training set statistics to all splits.
    """
    for feature in log_features:
        for df in [train_df, val_df, test_df]:
            x = df[feature]
            min_val = x.min()
            epsilon = 1e-12
            if min_val <= 0:
                shift = abs(min_val) + epsilon
                print(f"Shifting '{feature}' by {shift} before log transform.")
                df[feature] = np.log10(x + shift)
            else:
                df[feature] = np.log10(x)

    for feature in normal_features:
        mean = train_df[feature].mean()
        std = train_df[feature].std()
        for df in [train_df, val_df, test_df]:
            df[feature] = (mean - df[feature]) / std

    x_train = np.abs(train_df["cluster_time"]) ** (1 / 3) * np.sign(
        train_df["cluster_time"]
    )
    mean = x_train.mean()
    std = x_train.std()
    for df in [train_df, val_df, test_df]:
        x = np.abs(df["cluster_time"]) ** (1 / 3) * np.sign(df["cluster_time"])
        df["cluster_time"] = (x - mean) / std


def save_splits(train_df, val_df, test_df, base_name, tag):
    """
    Save each dataset split to an HDF5 file.
    """
    for split_name, split_df in zip(
        ["train", "val", "test"], [train_df, val_df, test_df]
    ):
        output_path = os.path.join(save_path, f"{base_name}{tag}_{split_name}.h5")
        with h5py.File(output_path, "w") as f:
            for col in split_df.columns:
                f.create_dataset(col, data=split_df[col].values)
        print(f"Saved {split_name} split to {output_path}")


# ---------- Main Function ---------- #
def main():
    """
    Entry point: parses args and triggers preprocessing in test or full mode.
    """
    apply_norm = not args.no_normalisation

    if args.test:
        print("Test mode activated...")
        tag = "mc20e"
        df_withpu = load_and_process(
            os.path.join(root_path, f"{tag}_withPU.root"),
            label=0,
            apply_norm=apply_norm,
        )
        df_nopu = load_and_process(
            os.path.join(root_path, f"{tag}_noPU.root"), label=1, apply_norm=apply_norm
        )
        df_combined = pd.concat([df_withpu, df_nopu], ignore_index=True)
        train_df, val_df, test_df = split_data(df_combined)
        if apply_norm:
            normalize_data(train_df, val_df, test_df)
            tag_suffix = "_norm"
        else:
            tag_suffix = "_raw"
        save_splits(train_df, val_df, test_df, tag, tag_suffix)

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
            df_combined = pd.concat([df_withpu, df_nopu], ignore_index=True)
            train_df, val_df, test_df = split_data(df_combined)
            if apply_norm:
                normalize_data(train_df, val_df, test_df)
                tag_suffix = "_norm"
            else:
                tag_suffix = "_raw"
            save_splits(train_df, val_df, test_df, tag, tag_suffix)


if __name__ == "__main__":
    main()
