"""
Preprocesses withPU and noPU .root files to hdf5 files with cuts applied.
"""

import os
import argparse
import numpy as np
import pandas as pd
import h5py
import uproot
import gc
import time
from datetime import datetime, timedelta
import psutil

from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

from config import (
    columns,
    log_features,
    normal_features,
    node_features,
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
    "--no_normalisation",
    action="store_true",
    help="Skip normalisation and time transformation.",
)
parser.add_argument(
    "--build_graphs",
    action="store_true",
    help="Build and save PyG graphs from normalized HDF5 splits.",
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


def to_cartesian(pt, eta, phi):
    """
    Transforms vector from ATLAS (spherical) coordinates to cartesian coordinates.
    """
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return np.stack((px, py, pz), axis=1)


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
        & (df["jetRawE"] > 0.0)
        & (df["jetRawPt"] > 0.0)
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


def compute_jet_features(df):
    """
    Computes zT, zL, zRel, and diffEta between cluster and jet using pt, eta, phi and adds them to df dictionary.
    Not needed cluster and jet features are dropped to keep hdf5 file size small.
    """
    # Compute cluster and jet cartesian pt vectors
    cluster_vec = to_cartesian(df["clusterPt"], df["clusterEta"], df["clusterPhi"])
    jet_vec = to_cartesian(df["jetRawPt"], df["jetRawEta"], df["jetRawPhi"])

    # Compute magnitude of cluster and jet vectors
    jet_mag2 = np.sum(jet_vec**2, axis=1)
    dot_product = np.sum(cluster_vec * jet_vec, axis=1)
    cross_product_mag = np.linalg.norm(np.cross(cluster_vec, jet_vec), axis=1)

    # Compute variables
    df["diffEta"] = df["clusterEta"] - df["jetRawEta"]
    df["zT"] = df["clusterPt"] / df["jetRawPt"]
    df["zL"] = dot_product / jet_mag2
    df["zRel"] = cross_product_mag / jet_mag2

    # Drop not needed cluster and jet features
    for key in ["jetRawEta", "jetRawPhi", "jetRawPt", "clusterPt"]:
        df.pop(key, None)
    return


def load_and_process(file_path, label, apply_norm):
    """
    Load ROOT file, apply cuts, compute response, and optionally apply high-pile-up cut.
    """
    print(f"Loading {file_path}...")
    tree = uproot.open(file_path)["ClusterTree;1"]
    df = tree.arrays(columns, library="np")
    length_before_cuts = len(df["clusterE"])

    # Apply physics cuts first
    mask = apply_cuts_mask(df)

    # High pile-up cut (only for background)
    if label == 0 and apply_norm:
        mask &= apply_high_pile_up_cut_mask(df)

    # Apply mask before computing response
    df = {key: val[mask] for key, val in df.items()}

    # Now compute response
    response_mask = compute_response_and_mask(df)
    df = {key: val[response_mask] for key, val in df.items()}
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
    Applies log10 scaling to log_features, and standard scaling to normal_features.
    Also applies cube-root normalization to cluster_time.
    """
    stats_lines = [f"Normalization statistics for {tag}\n"]
    epsilon = 1e-6

    # Apply log10 scaling ONLY for log_features
    for feature in log_features:
        for split in [train, val, test]:
            x = split[feature]
            x = np.where(x > 0, x, epsilon)  # Replace non-positive with epsilon
            x = np.log10(x)
            x[~np.isfinite(x)] = 0  # Replace any -inf/nan with 0
            split[feature] = x
        stats_lines.append(f"{feature}: log10 scaled (replaced <=0 with {epsilon})\n")

    # Apply standard scaling for normal_features
    for feature in normal_features:
        finite_mask = np.isfinite(train[feature])
        if not np.any(finite_mask):
            print(f"Warning: No finite values for {feature}. Skipping normalization.")
            continue
        mean = train[feature][finite_mask].mean()
        std = train[feature][finite_mask].std()
        for split in [train, val, test]:
            x = split[feature]
            x = (x - mean) / std
            x[~np.isfinite(x)] = 0
            split[feature] = x
        stats_lines.append(f"{feature}: mean = {mean:.6f}, std = {std:.6f}\n")

    # Normalize cluster_time with cube-root scaling
    x_train_time = np.cbrt(np.abs(train["cluster_time"])) * np.sign(
        train["cluster_time"]
    )
    finite_mask = np.isfinite(x_train_time)
    mean = x_train_time[finite_mask].mean()
    std = x_train_time[finite_mask].std()

    for split in [train, val, test]:
        x = np.cbrt(np.abs(split["cluster_time"])) * np.sign(split["cluster_time"])
        x = (x - mean) / std
        x[~np.isfinite(x)] = 0
        split["cluster_time"] = x

    stats_lines.append(f"cluster_time (cbrt): mean = {mean:.6f}, std = {std:.6f}\n")

    # Save statistics
    stats_path = os.path.join(save_path, f"{tag}_norm_stats.txt")
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


def load_multiple_campaigns(campaigns, apply_norm):
    """
    Loads and concatenates multiple sub-campaigns (e.g., mc20a, mc20d, mc20e).
    """
    combined_data = {}
    for tag in campaigns:
        print(f"Loading sub-campaign: {tag}")
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
        df_combined = {
            key: np.concatenate([df_withpu[key], df_nopu[key]])
            for key in df_withpu.keys()
        }
        if not combined_data:
            combined_data = {k: [v] for k, v in df_combined.items()}
        else:
            for k in combined_data:
                combined_data[k].append(df_combined[k])
    # Final merge
    return {k: np.concatenate(v) for k, v in combined_data.items()}


def get_memory_usage():
    process = psutil.Process(os.getpid())
    ram_bytes = process.memory_info().rss
    ram_gb = ram_bytes / 1e9
    vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    return ram_gb, vram_gb


def build_jetwise_graphs_with_edges(h5_path, feature_keys, max_batch_graphs=100000):
    """
    Builds jet-wise PyG graphs in batches using NumPy (no pandas),
    with RAM, VRAM and time monitoring.
    """
    print(
        f"[{datetime.now()}] Building graphs from {h5_path} with RAM/VRAM/time monitoring..."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    with h5py.File(h5_path, "r") as f:
        data = {k: f[k][:] for k in f.keys()}

    event_numbers = data["eventNumber"]
    jet_counts = data["jetCnt"]
    unique_keys, inverse = np.unique(
        np.stack((event_numbers, jet_counts), axis=1), axis=0, return_inverse=True
    )

    total_jets = len(unique_keys)
    all_batches = []
    current_batch = []

    for jet_id in range(total_jets):
        indices = np.where(inverse == jet_id)[0]
        if len(indices) < 2:
            continue

        x_np = np.stack([data[feat][indices] for feat in feature_keys], axis=1).astype(
            np.float32
        )
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(data["label"][indices]).to(
            dtype=torch.float32, device=device
        )

        num_nodes = x.size(0)
        edge_index = (
            torch.combinations(torch.arange(num_nodes, device=device), r=2)
            .t()
            .contiguous()
        )
        edge_attr = torch.cdist(x, x, p=1)[edge_index[0], edge_index[1]].unsqueeze(1)

        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        num_signal = (y == 1).sum().item()
        num_pu = (y == 0).sum().item()
        pu_weight = num_signal / num_pu if num_signal > 0 and num_pu > 0 else 1.0
        weights = torch.where(
            y == 0,
            torch.tensor(pu_weight, device=device),
            torch.tensor(1.0, device=device),
        )

        event = event_numbers[indices[0]]
        jet = jet_counts[indices[0]]

        graph = Data(
            x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, weights=weights
        )
        graph.eventNumber = torch.tensor(event, device=device)
        graph.jetCnt = torch.tensor(jet, device=device)

        current_batch.append(graph)

        if len(current_batch) >= max_batch_graphs:
            all_batches.append(current_batch)
            current_batch = []
            torch.cuda.empty_cache()
            gc.collect()

        if jet_id % 10000 == 0:
            elapsed = time.time() - start_time
            ram, vram = get_memory_usage()
            print(
                f"[✓] Processed jets {jet_id}/{total_jets} | RAM: {ram:.2f} GB | VRAM: {vram:.2f} GB | Time: {str(timedelta(seconds=int(elapsed)))}"
            )

    if current_batch:
        all_batches.append(current_batch)

    print(f"[✓] Finished building {total_jets} jets in {len(all_batches)} batches")
    return all_batches


def build_and_save_jetwise_graphs(tag, feature_keys):
    """
    Builds jetwise graphs with edges and stores them as pt files.
    """
    for split in ["train", "val", "test"]:
        h5_path = os.path.join(save_path, f"{tag}_norm_{split}.h5")
        if not os.path.exists(h5_path):
            print(f"  [!] File not found: {h5_path}, skipping.")
            continue
        graphs = build_jetwise_graphs_with_edges(h5_path, feature_keys)
        out_path = os.path.join(save_path, f"{tag}_graphs_{split}.pt")
        torch.save(graphs, out_path)
        print(f"  -> Saved graphs to {out_path}")


# ---------- Main ---------- #
def main():
    if args.print_features:
        list_root_features(os.path.join(root_path, "mc20a_withPU.root"))
        return

    if args.build_graphs:
        if args.campaign:
            build_and_save_jetwise_graphs(args.campaign, node_features)
        elif args.full:
            for tag in ["mc20a", "mc20d", "mc20e", "mc23a", "mc23d", "mc23e"]:
                build_and_save_jetwise_graphs(tag, node_features)
        else:
            print("Error: --build_graphs requires --campaign or --full")
        return

    apply_norm = not args.no_normalisation

    if args.campaign:
        tag = args.campaign

        if not apply_norm:
            print(f"Processing raw mode for campaign: {tag}")

            for label_name, label in [("withPU", 0), ("noPU", 1)]:
                df = load_and_process(
                    os.path.join(root_path, f"{tag}_{label_name}.root"),
                    label=label,
                    apply_norm=False,
                )
                compute_jet_features(df)
                save_split(df, tag, f"{label_name}_raw")
            return

        if tag in ["mc20", "mc23"]:
            sub_tags = {
                "mc20": ["mc20a", "mc20d", "mc20e"],
                "mc23": ["mc23a", "mc23d", "mc23e"],
            }[tag]
            print(f"Preprocessing combined campaign: {tag}")
            combined = load_multiple_campaigns(sub_tags, apply_norm=True)
        else:
            print(f"Processing individual campaign: {tag}")
            df_withpu = load_and_process(
                os.path.join(root_path, f"{tag}_withPU.root"),
                label=0,
                apply_norm=True,
            )
            df_nopu = load_and_process(
                os.path.join(root_path, f"{tag}_noPU.root"),
                label=1,
                apply_norm=True,
            )
            combined = {
                key: np.concatenate([df_withpu[key], df_nopu[key]])
                for key in df_withpu.keys()
            }

        train, val, test = split_data_full(combined)
        normalize_data(train, val, test, tag)

        compute_jet_features(train)
        compute_jet_features(val)
        compute_jet_features(test)

        save_split(train, tag, "norm_train")
        save_split(val, tag, "norm_val")
        save_split(test, tag, "norm_test")
        return

    if args.full:
        print("Full mode activated...")
        tags = ["mc20a", "mc20d", "mc20e", "mc23a", "mc23d", "mc23e"]
        for tag in tags:
            print(f"Processing {tag}...")

            if not apply_norm:
                for label_name, label in [("withPU", 0), ("noPU", 1)]:
                    df = load_and_process(
                        os.path.join(root_path, f"{tag}_{label_name}.root"),
                        label=label,
                        apply_norm=False,
                    )
                    compute_jet_features(df)
                    save_split(df, tag, f"{label_name}_raw")
                continue

            df_withpu = load_and_process(
                os.path.join(root_path, f"{tag}_withPU.root"),
                label=0,
                apply_norm=True,
            )
            df_nopu = load_and_process(
                os.path.join(root_path, f"{tag}_noPU.root"),
                label=1,
                apply_norm=True,
            )
            combined = {
                key: np.concatenate([df_withpu[key], df_nopu[key]])
                for key in df_withpu.keys()
            }

            train, val, test = split_data_full(combined)
            normalize_data(train, val, test, tag)

            compute_jet_features(train)
            compute_jet_features(val)
            compute_jet_features(test)

            save_split(train, tag, "norm_train")
            save_split(val, tag, "norm_val")
            save_split(test, tag, "norm_test")


if __name__ == "__main__":
    main()
