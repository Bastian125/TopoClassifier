"""
Dataloader that either loads MC20a as Run 2a, or Run 2 (MC20a, MC20d, MC20e) or Run3 (MC23a, MC23d, MC23e).
"""

# ---------- Imports ---------- #
import os
import argparse

import h5py
import numpy as np

import tensorflow as tf
from sklearn.model_selection import train_test_split


# ---------- Data Loader Class ---------- #
class HDF5DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        file_paths,
        feature_keys,
        batch_size=512,
        shuffle=True,
        mode="train",
        val_split=0.2,
        random_state=42,
    ):
        self.file_paths = file_paths
        self.feature_keys = feature_keys
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.val_split = val_split
        self.random_state = random_state
        self.data, self.labels = self._load_and_split_data()
        self.input_dim = self.data.shape[1]
        self.indices = np.arange(len(self.labels))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _load_data_from_file(self, path):
        with h5py.File(path, "r") as f:
            X = np.stack([f[key][:] for key in self.feature_keys], axis=1)
            y = f["label"][:]  # Load the 'label' dataset from the file
        return X, y

    def _load_and_split_data(self):
        X_all, y_all = [], []
        for path in self.file_paths:
            X, y = self._load_data_from_file(path)
            X_all.append(X)
            y_all.append(y)
        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)

        # Stratified split
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X_all,
            y_all,
            test_size=self.val_split,
            stratify=y_all,
            random_state=self.random_state,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval,
            y_trainval,
            test_size=self.val_split,
            stratify=y_trainval,
            random_state=self.random_state,
        )

        if self.mode == "train":
            return X_train, y_train
        elif self.mode == "val":
            return X_val, y_val
        elif self.mode == "test":
            return X_test, y_test
        else:
            raise ValueError("mode must be 'train', 'val', or 'test'")

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
