import h5py
import numpy as np
import glob
import os

import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Data


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


class GraphBatchIterableDataset(IterableDataset):
    def __init__(self, pt_file_pattern):
        """
        Args:
            pt_file_pattern (str): Glob pattern like '/path/to/mc20e_graphs_train_batch_*.pt'
        """
        self.pt_file_pattern = pt_file_pattern
        self.files = sorted(glob.glob(pt_file_pattern))
        if not self.files:
            raise FileNotFoundError(
                f"No .pt files found matching pattern: {pt_file_pattern}"
            )

    def __iter__(self):
        for pt_file in self.files:
            graphs = torch.load(pt_file)
            for graph in graphs:
                yield graph
