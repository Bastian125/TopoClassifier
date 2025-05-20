import h5py
import numpy as np

import torch
import tqdm
from torch.utils.data import Dataset


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
