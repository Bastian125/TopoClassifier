# ---------- Imports ---------- #
import h5py
import numpy as np
import glob
import os

import torch
from torch.utils.data import Dataset, IterableDataset
from torch_geometric.loader import DataLoader as GeoDataLoader


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


class JetGraphIterableDataset(IterableDataset):
    def __init__(self, file_pattern_or_list, shuffle_files=False):
        if isinstance(file_pattern_or_list, str):
            self.files = sorted(glob.glob(file_pattern_or_list))
        else:
            self.files = sorted(file_pattern_or_list)
        self.shuffle_files = shuffle_files

    def __iter__(self):
        file_list = self.files.copy()
        if self.shuffle_files:
            import random

            random.shuffle(file_list)

        for file_path in file_list:
            try:
                data_list = torch.load(file_path, map_location="cpu")
                for graph in data_list:
                    yield graph
            except Exception as e:
                print(f"Skipping {file_path} due to error: {e}")
