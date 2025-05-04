"""
Dataloader that either loads MC20a as Run2a, or Run2d, Run2e or Run3a, Run3d, Run3e.
Updated to use pre-split HDF5 files: *_train.h5, *_val.h5, *_test.h5.
"""

# ---------- Imports ---------- #
import os
import h5py
import numpy as np
import tensorflow as tf


# ---------- Data Loader Class ---------- #
class HDF5DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        file_path,
        feature_keys,
        batch_size=512,
        shuffle=True,
    ):
        self.file_path = file_path
        self.feature_keys = feature_keys
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data, self.labels = self._load_data()
        self.input_dim = self.data.shape[1]
        self.indices = np.arange(len(self.labels))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _load_data(self):
        with h5py.File(self.file_path, "r") as f:
            X = np.stack([f[key][:] for key in self.feature_keys], axis=1)
            y = f["label"][:]
        return X, y

    def __len__(self):
        return int(np.ceil(len(self.labels) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        X_batch = self.data[batch_indices]
        y_batch = self.labels[batch_indices]
        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
