"""
Preprocesses withPU and noPU .root files to hdf5 files with cuts applied.
"""

# ---------- Imports ---------- #
import os
import argparse
import numpy as np
import pandas as pd
import h5py
import uproot
from datetime import datetime
from tqdm import tqdm
import psutil
import glob
import gc

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch_geometric.data import Data

from config import (
    columns,
    log_features,
    normal_features,
    data_root_path as root_path,
    data_save_path as save_path,
    jet_feature_keys as node_features,
)

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
    "--prepare_graphs",
    action="store_true",
    help="Prepare HDF5 files for graph building.",
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


def save_split(df, base_name, tag, pos_weight=None):
    """
    Save a dictionary of numpy arrays as an HDF5 file.
    """
    output_path = os.path.join(save_path, f"{base_name}_{tag}.h5")
    with h5py.File(output_path, "w") as f:
        for key, val in df.items():
            f.create_dataset(key, data=val)
        if pos_weight is not None:
            f.attrs["pos_weight"] = pos_weight
    print(f"Saved {tag} split to {output_path}")

    del df
    gc.collect()


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


def print_memory():
    """
    Prints current memory usage in GB.
    """
    ram = psutil.Process().memory_info().rss / 1e9
    print(f"[RAM] Current usage: {ram:.2f} GB")


def build_graphs(
    h5_path, feature_keys, output_path, chunk_size=100000, merge_chunks=False
):
    """
    Builds and saves PyTorch Geometric graphs from HDF5 into chunked .pt files.
    Then optionally merges them into a final single .pt file.
    """
    print(f"[{datetime.now()}] Loading HDF5 into pandas DataFrame from {h5_path}...")
    with h5py.File(h5_path, "r") as f:
        data = {k: f[k][:] for k in f.keys()}
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} entries")

    grouped = df.groupby(["eventNumber", "jetCnt"])
    total_jets = len(grouped)
    print(f"[INFO] Total jets to process: {total_jets}")

    graph_list = []
    chunk_idx = 0

    for idx, ((event, jet), group) in enumerate(
        tqdm(grouped, desc="Building graphs", unit="jet")
    ):
        if len(group) < 2:
            continue

        x_np = group[feature_keys].to_numpy(dtype=np.float32)
        y_np = group["label"].to_numpy(dtype=np.float32)
        response_np = group["cluster_response"].to_numpy(dtype=np.float32)

        if x_np.shape[0] != y_np.shape[0] or x_np.ndim != 2:
            continue

        x = torch.tensor(x_np, dtype=torch.float32)
        y = torch.tensor(y_np, dtype=torch.float32)

        num_nodes = x.size(0)
        node_indices = torch.arange(num_nodes)
        edge_index = torch.combinations(node_indices, r=2).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        graph = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            cluster_response=torch.tensor(response_np, dtype=torch.float32),
            eventNumber=torch.full((num_nodes,), event, dtype=torch.int32),
            jetCnt=torch.full((num_nodes,), jet, dtype=torch.int32),
        )

        graph_list.append(graph)

        if len(graph_list) >= chunk_size:
            chunk_path = output_path.replace(".pt", f"_chunk{chunk_idx:03d}.pt")
            torch.save(graph_list, chunk_path)
            print(
                f"[{datetime.now()}] Saved chunk {chunk_idx} with {len(graph_list)} graphs → {chunk_path}"
            )
            print_memory()
            graph_list.clear()
            chunk_idx += 1

    if graph_list:
        chunk_path = output_path.replace(".pt", f"_chunk{chunk_idx:03d}.pt")
        torch.save(graph_list, chunk_path)
        print(
            f"[{datetime.now()}] Saved final chunk {chunk_idx} with {len(graph_list)} graphs → {chunk_path}"
        )
        print_memory()

    print(f"[✓] All chunks saved.")

    # Optional merge
    if merge_chunks:
        print(f"[INFO] Merging all chunks into single file...")
        chunk_files = sorted(glob.glob(output_path.replace(".pt", "_chunk*.pt")))
        merged_graphs = []
        for file in tqdm(chunk_files, desc="Merging chunks", unit="file"):
            merged_graphs.extend(torch.load(file))

        torch.save(merged_graphs, output_path)
        print(f"Merged {len(merged_graphs)} graphs to: {output_path}")


def build_and_save_jetwise_graphs(tag, feature_keys):
    for split in ["train", "val", "test"]:
        h5_path = os.path.join(save_path, f"{tag}_graph_{split}.h5")
        output_path = os.path.join(save_path, f"{tag}_graph_{split}.pt")

        print(f"\n[INFO] Processing split: {split.upper()}")
        if not os.path.exists(h5_path):
            print(f"  [!] File not found: {h5_path}, skipping.")
            continue

        print_memory()

        build_graphs(h5_path, feature_keys, output_path, chunk_size=100000)

        gc.collect()

        print_memory()

        print(f"Saved graphs to: {output_path}")


def stratified_jetwise_split(df, train_frac=0.6, val_frac=0.2, test_frac=0.2):
    assert (
        abs(train_frac + val_frac + test_frac - 1.0) < 1e-5
    ), "Fractions must sum to 1."

    # Group jets by (eventNumber, jetCnt)
    df_all = pd.DataFrame(df)
    df_all["jet_id"] = list(zip(df_all["eventNumber"], df_all["jetCnt"]))

    # Assign one label per jet via majority vote
    jet_labels = df_all.groupby("jet_id")["label"].mean().round().astype(int)

    jet_ids = np.array(jet_labels.index.tolist())
    jet_y = jet_labels.values

    # Stratified jet-level splitting
    jet_ids_trainval, jet_ids_test = train_test_split(
        jet_ids, stratify=jet_y, test_size=test_frac, random_state=42
    )
    jet_labels_trainval = jet_labels.loc[list(map(tuple, jet_ids_trainval))]
    jet_ids_train, jet_ids_val = train_test_split(
        jet_ids_trainval,
        stratify=jet_labels_trainval,
        test_size=val_frac / (train_frac + val_frac),
        random_state=42,
    )

    # Build boolean masks for each jet set
    def mask_from_jet_ids(jet_ids_set):
        jet_set = set(map(tuple, jet_ids_set))
        return np.array([tuple(ev_jet) in jet_set for ev_jet in df_all["jet_id"]])

    train_mask = mask_from_jet_ids(jet_ids_train)
    val_mask = mask_from_jet_ids(jet_ids_val)
    test_mask = mask_from_jet_ids(jet_ids_test)

    df_train = {k: v[train_mask] for k, v in df.items()}
    df_val = {k: v[val_mask] for k, v in df.items()}
    df_test = {k: v[test_mask] for k, v in df.items()}

    return df_train, df_val, df_test


def process_campaign(tag):
    print(f"[INFO] Preparing graph-ready HDF5 for {tag}")
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
        key: np.concatenate([df_withpu[key], df_nopu[key]]) for key in df_withpu.keys()
    }

    del df_withpu, df_nopu
    gc.collect()

    # Sort by eventNumber and jetCnt
    sort_index = np.lexsort((combined["jetCnt"], combined["eventNumber"]))
    for key in combined:
        combined[key] = combined[key][sort_index]

    # Stratified jet-wise split
    train, val, test = stratified_jetwise_split(combined)

    del combined
    gc.collect()

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1]), y=train["label"]
    )
    pos_weight = class_weights[1] / class_weights[0]

    normalize_data(train, val, test, tag)

    compute_jet_features(train)
    compute_jet_features(val)
    compute_jet_features(test)

    save_split(train, tag, "graph_train", pos_weight=pos_weight)
    save_split(val, tag, "graph_val", pos_weight=pos_weight)
    save_split(test, tag, "graph_test", pos_weight=pos_weight)


# ---------- Main ---------- #
def main():
    if args.print_features:
        list_root_features(os.path.join(root_path, "mc20a_withPU.root"))
        return

    if args.prepare_graphs:
        if args.campaign:
            tag = args.campaign
            if tag in ["mc20", "mc23"]:
                sub_tags = {
                    "mc20": ["mc20a", "mc20d", "mc20e"],
                    "mc23": ["mc23a", "mc23d", "mc23e"],
                }[tag]
                for sub_tag in sub_tags:
                    process_campaign(sub_tag)
            else:
                process_campaign(tag)
        elif args.full:
            tags = ["mc20a", "mc20d", "mc20e", "mc23a", "mc23d", "mc23e"]
            for tag in tags:
                process_campaign(tag)
        else:
            print("Error: --prepare_graphs requires --campaign or --full")
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

                del df
                gc.collect()
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

        # Compute class weights
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.array([0, 1]), y=train["label"]
        )
        pos_weight = class_weights[1] / class_weights[0]

        normalize_data(train, val, test, tag)

        compute_jet_features(train)
        compute_jet_features(val)
        compute_jet_features(test)

        save_split(train, tag, "norm_train", pos_weight=pos_weight)
        save_split(val, tag, "norm_val", pos_weight=pos_weight)
        save_split(test, tag, "norm_test", pos_weight=pos_weight)
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

            # Compute class weights
            class_weights = compute_class_weight(
                class_weight="balanced", classes=np.array([0, 1]), y=train["label"]
            )
            pos_weight = class_weights[1] / class_weights[0]

            normalize_data(train, val, test, tag)

            compute_jet_features(train)
            compute_jet_features(val)
            compute_jet_features(test)

            save_split(train, tag, "norm_train", pos_weight=pos_weight)
            save_split(val, tag, "norm_val", pos_weight=pos_weight)
            save_split(test, tag, "norm_test", pos_weight=pos_weight)


if __name__ == "__main__":
    main()
