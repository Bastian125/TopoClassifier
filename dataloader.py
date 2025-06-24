# ---------- Imports ---------- #
import h5py
import numpy as np
import glob
import re
import time

import torch
from torch.utils.data import Dataset, IterableDataset
from torch_geometric.data import Data


def natural_sort_key(path):
    return [int(text) if text.isdigit() else text for text in re.split(r"(\d+)", path)]


class HDF5Dataset(Dataset):
    """
    PyTorch dataset for loading normalized ATLAS features from an HDF5 file.
    Each sample includes a set of features and a binary label.
    """

    def __init__(self, file_path, feature_keys, target):
        self.file_path = file_path
        self.feature_keys = feature_keys

        # Load entire dataset into memory
        with h5py.File(self.file_path, "r") as f:
            self.data = torch.tensor(
                np.stack([f[key][:] for key in self.feature_keys], axis=1),
                dtype=torch.float32,
            )
            self.labels = torch.tensor(f[target][:], dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def get_target_array(self, target_key="cluster_response"):
        """
        Utility to extract full array of a given target from the HDF5 file.
        """
        with h5py.File(self.file_path, "r") as f:
            return np.array(f[target_key])

    def get_all_data(self):
        """
        Returns the full feature matrix and label vector as NumPy arrays.
        Useful for analysis like permutation importance.
        """
        return self.data.numpy(), self.labels.numpy()


class LazyGraphDataset(Dataset):
    def __init__(self, file_pattern, chunk_size=100_000):
        self.chunk_paths = sorted(glob.glob(file_pattern), key=natural_sort_key)
        self.index_map = []
        self._cache = {}

        for chunk_idx, path in enumerate(self.chunk_paths):
            print(f"[INFO] Loading chunk metadata from: {path}")
            start = time.time()

            if chunk_idx < len(self.chunk_paths) - 1:
                num_graphs = chunk_size
            else:
                # Only load the last chunk to get its actual size
                graphs = torch.load(path, map_location="cpu")
                num_graphs = len(graphs)
                print(
                    f"[✓] Loaded {num_graphs} graphs from last chunk in {time.time() - start:.2f} sec"
                )

            for i in range(num_graphs):
                self.index_map.append((chunk_idx, i))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        chunk_idx, graph_idx = self.index_map[idx]
        print(f"[DEBUG] Accessing graph {graph_idx} from chunk {chunk_idx}")

        if chunk_idx not in self._cache:
            self._cache = {}  # clear cache
            path = self.chunk_paths[chunk_idx]
            print(f"[DEBUG] Loading chunk {chunk_idx} from {path}")
            self._cache[chunk_idx] = torch.load(path, map_location="cpu")

        return self._cache[chunk_idx][graph_idx]
