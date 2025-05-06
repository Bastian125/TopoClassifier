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
    Apply consistent physics-motivated cuts on calorimeter cluster variables.
    Removes entries with unphysical or undefined values.
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
    Apply avgMu > 20 to select pile-up dominated events (used only for background).
    """
    return df[(df["avgMu"] > 20)]


def compute_response(df):
    """
    Compute response as clusterE / cluster_ENG_CALIB_TOT.
    Apply a cut to keep only entries with response > 0.1.
    """
    df["cluster_response"] = df["clusterE"] / df["cluster_ENG_CALIB_TOT"]
    return df[df["cluster_response"] > 0.1]


def load_and_process(file_path, label, apply_norm):
    """
    Load ROOT file and return a preprocessed pandas DataFrame with:
    - cuts applied
    - label assigned (0 for PU, 1 for signal)
    - response calculated and filtered
    """
    print(f"Loading {file_path}...")
    root_file = uproot.open(file_path)
    tree = root_file["ClusterTree;1"]
    df = tree.arrays(columns, library="pd")
    df = apply_cuts(df)
    if label == 0 and apply_norm:
        df = apply_high_pile_up_cut(df)
    df["label"] = label
    df = compute_response(df)
    print(f"  -> {len(df)} entries retained after all cuts\n")
    return df


def split_data_full(df):
    """
    Stratified split into 60% train, 20% val, 20% test.
    """
    trainval_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    train_df, val_df = train_test_split(
        trainval_df, test_size=0.25, stratify=trainval_df["label"], random_state=42
    )
    return train_df, val_df, test_df


def normalize_data(train_df, val_df, test_df):
    """
    Apply log and standard normalization using statistics from the training set.
    Also applies cubic root transformation to cluster_time.
    """
    for feature in log_features:
        x_train = train_df[feature]
        min_val = x_train.min()
        epsilon = 1e-12
        shift = abs(min_val) + epsilon if min_val <= 0 else 0
        train_df[feature] = np.log10(x_train + shift)
        val_df[feature] = np.log10(val_df[feature] + shift)
        test_df[feature] = np.log10(test_df[feature] + shift)

    for feature in normal_features:
        mean = train_df[feature].mean()
        std = train_df[feature].std()
        train_df[feature] = (train_df[feature] - mean) / std
        val_df[feature] = (val_df[feature] - mean) / std
        test_df[feature] = (test_df[feature] - mean) / std

    x_train_time = np.abs(train_df["cluster_time"]) ** (1 / 3) * np.sign(
        train_df["cluster_time"]
    )
    mean = x_train_time.mean()
    std = x_train_time.std()

    for df in [train_df, val_df, test_df]:
        x = np.abs(df["cluster_time"]) ** (1 / 3) * np.sign(df["cluster_time"])
        df["cluster_time"] = (x - mean) / std


def save_split(df, base_name, tag):
    """
    Save a single DataFrame to an HDF5 file under data_save_path.
    """
    output_path = os.path.join(save_path, f"{base_name}_{tag}.h5")
    with h5py.File(output_path, "w") as f:
        for col in df.columns:
            f.create_dataset(col, data=df[col].values)
    print(f"Saved {tag} split to {output_path}\n")


# ---------- Main Function ---------- #
def main():
    """
    Main entry point for preprocessing. Handles test or full mode.
    Loads ROOT files, applies cuts and normalization, and saves HDF5 splits.
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
        train_df, val_df, test_df = split_data_full(df_combined)
        if apply_norm:
            normalize_data(train_df, val_df, test_df)
            tag_suffix = "norm"
        else:
            tag_suffix = "raw"
        save_split(train_df, tag, f"{tag_suffix}_train")
        save_split(val_df, tag, f"{tag_suffix}_val")
        save_split(test_df, tag, f"{tag_suffix}_test")

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
            train_df, val_df, test_df = split_data_full(df_combined)
            if apply_norm:
                normalize_data(train_df, val_df, test_df)
                tag_suffix = "norm"
            else:
                tag_suffix = "raw"
            save_split(train_df, tag, f"{tag_suffix}_train")
            save_split(val_df, tag, f"{tag_suffix}_val")
            save_split(test_df, tag, f"{tag_suffix}_test")


if __name__ == "__main__":
    main()
