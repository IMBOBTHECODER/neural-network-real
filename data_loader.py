import numpy as np
import pandas as pd
import pickle
import os
try:
    import cupy as xp
except ImportError:
    import numpy as xp


class DataLoader:
    def __init__(self, X, Y, batch_size=64, shuffle=True):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_classes = int(xp.max(Y).item()) + 1
        self.Y_onehot = xp.eye(self.num_classes, dtype=xp.float32)[Y]   # computed once

    def __iter__(self):
        indices = xp.arange(len(self.X))
        if self.shuffle:
            xp.random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start:start + self.batch_size]
            yield self.X[batch], self.Y_onehot[batch]   # just slice, no recompute

    def __len__(self):
        return (len(self.X) + self.batch_size - 1) // self.batch_size


def get_emnist_data(train_path, test_path):
    """
    Loads EMNIST from Kaggle-style CSVs: label in column 0, 784 flat pixel
    columns after it, one row per image. Returns flat (N, 784) float32 arrays,
    normalized to [0, 1].
    """
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    Y_train = train_df.iloc[:, 0].to_numpy().astype(np.int32)
    X_train = train_df.iloc[:, 1:].to_numpy().astype(np.float32) / 255.0

    Y_test = test_df.iloc[:, 0].to_numpy().astype(np.int32)
    X_test = test_df.iloc[:, 1:].to_numpy().astype(np.float32) / 255.0

    X_train, X_test = xp.asarray(X_train), xp.asarray(X_test)
    Y_train, Y_test = xp.asarray(Y_train), xp.asarray(Y_test)

    return X_train, X_test, Y_train, Y_test


def _load_cifar10_batch(file_path):
    """
    Loads one CIFAR-10 pickle batch file (the standard distribution format:
    'cifar-10-batches-py/data_batch_1' ... 'data_batch_5', 'test_batch').
    Each batch stores 'data' as (10000, 3072) uint8 — flattened, channel-major
    (all 1024 red values, then all 1024 green, then all 1024 blue, per image).
    """
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data']            # (10000, 3072), uint8
    labels = batch[b'labels']        # list of 10000 ints
    return data, labels


def get_cifar10_data(data_dir):
    """
    Loads CIFAR-10 from the standard 'cifar-10-batches-py' directory structure
    (5 training batch files + 1 test batch file, as distributed by the
    official CIFAR-10 release and most Kaggle mirrors of it).

    Returns FLAT (N, 3072) float32 arrays, normalized to [0, 1], in the same
    "channel-major flattened" order the raw files use — matching how your
    CNN.forward reshapes flat input back to (C, H, W).
    """
    train_data_list = []
    train_labels_list = []

    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f"data_batch_{i}")
        data, labels = _load_cifar10_batch(batch_path)
        train_data_list.append(data)
        train_labels_list.extend(labels)

    X_train = np.concatenate(train_data_list, axis=0).astype(np.float32) / 255.0
    Y_train = np.array(train_labels_list, dtype=np.int32)

    test_path = os.path.join(data_dir, "test_batch")
    test_data, test_labels = _load_cifar10_batch(test_path)
    X_test = test_data.astype(np.float32) / 255.0
    Y_test = np.array(test_labels, dtype=np.int32)

    X_train, X_test = xp.asarray(X_train), xp.asarray(X_test)
    Y_train, Y_test = xp.asarray(Y_train), xp.asarray(Y_test)

    return X_train, X_test, Y_train, Y_test