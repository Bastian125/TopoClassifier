"""
Preprocess root files for training and store them as npy.
"""

# ---------- Imports ---------- #
import os
import argparse

import uproot
import numpy as np
import pandas as pd

from config import columns, log_features, normal_features, root_path, save_path

# ---------- Argument Parser ---------- #
parser = argparse.ArgumentParser(description="Perform preprocessing of root files.")
mode_group = parser.add_mutually_exclusive_group(required=True)
mode_group.add_argument("--test", action="store_true", help="Run in test mode (process only mc20a_withPU)")
mode_group.add_argument("--full", action="store_true", help="Run full preprocessing on all datasets")
parser.add_argument("--no-normalisation", action="store_true", help="Skip normalisation and time transformation")
args = parser.parse_args()

# ---------- Helper Functions ---------- #
def apply_cuts(df):
    df.drop(df[df["clusterE"] <= 0].index, inplace=True)
    df.drop(df[df["cluster_ENG_CALIB_TOT"] <= 0.3].index, inplace=True)
    df.drop(df[df["cluster_CENTER_LAMBDA"] <= 0.0].index, inplace=True)
    df.drop(df[df["cluster_FIRST_ENG_DENS"] <= 0.0].index, inplace=True)
    df.drop(df[df["cluster_SECOND_TIME"] <= 0.0].index, inplace=True)
    df.drop(df[df["cluster_SIGNIFICANCE"] <= 0.0].index, inplace=True)
    df.drop("cluster_SIGNIFICANCE", axis=1, inplace=True)

def compute_response(df):
    df["cluster_response"] = df["clusterE"] / df["cluster_ENG_CALIB_TOT"]
    df.drop("cluster_ENG_CALIB_TOT", axis=1, inplace=True)

def apply_log(df, feature):
    x = df[feature]
    min_val = x.min()
    epsilon = 1e-12
    if min_val <= 0:
        shift = abs(min_val) + epsilon
        print(f"Shifting '{feature}' by {shift} before log transform to avoid non-positive values.")
        df[feature] = np.log10(x + shift)
    else:
        df[feature] = np.log10(x)

def apply_normalisation(df, feature):
    x = df[feature]
    df[feature] = (x.mean() - x) / x.std()

def apply_time_normalisation(df):
    x = df["cluster_time"]
    transformed = np.abs(x) ** (1 / 3) * np.sign(x)
    df["cluster_time"] = (transformed - transformed.mean()) / transformed.std()

def preprocess_root_file(file_path, output_base_name, apply_norm=True):
    print(f"Preprocessing: {file_path}")
    root_file = uproot.open(file_path)
    tree = root_file["ClusterTree;1"]
    df = tree.arrays(columns, library="pd")
    print("Data loaded...")

    apply_cuts(df)
    print("Cuts applied...")

    compute_response(df)
    print("Response computed...")

    tag = "_norm" if apply_norm else "_raw"

    if apply_norm:
        for feature in log_features:
            apply_log(df, feature)
        print("Log transformation applied...")

        for feature in normal_features:
            apply_normalisation(df, feature)
        print("Normalization applied...")

        apply_time_normalisation(df)
        print("Special time normalization applied...")
    else:
        print("Skipping log scale, normalization and time transformation.")

    os.makedirs(save_path, exist_ok=True)
    output_name = f"{output_base_name.replace('.npy','')}{tag}.npy"
    output_path = os.path.join(save_path, output_name)
    np.save(output_path, df.to_numpy())
    print(f"Saved preprocessed data to {output_path}\n")

# ---------- Main Function ---------- #
def main():
    apply_norm = not args.no_normalisation

    if args.test:
        print("Test mode activated...")
        preprocess_root_file(os.path.join(root_path, "mc20a_withPU.root"), "mc20a_withPU.npy", apply_norm=apply_norm)
    elif args.full:
        print("Full mode activated...")
        for tag in ["mc20a", "mc20d", "mc20e", "mc23a", "mc23d", "mc23e"]:
            for pu in ["withPU", "noPU"]:
                file_name = f"{tag}_{pu}.root"
                output_name = f"{tag}_{pu}.npy"
                preprocess_root_file(os.path.join(root_path, file_name), output_name, apply_norm=apply_norm)

if __name__ == "__main__":
    main()
