"""
data/dataset.py
================
Carga las features cacheadas (extraídas con los ONNX reales de IntellEvent)
y arma tensores PyTorch con padding + máscara, listos para entrenar la
cabeza de clasificación de PC.

Cada muestra: features (T, 4) = [P(no_IC), P(IC), P(no_FO), P(FO)] por frame,
              label = clase Ferrari (0-3)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG


def load_cached_dataset(cache_dir, max_seq_len):
    """
    Carga todas las features cacheadas, recorta/rellena a max_seq_len,
    y arma los arrays finales para el Dataset.

    Returns:
        X: np.ndarray (N, max_seq_len, 4)
        y: np.ndarray (N,)
        mask: np.ndarray (N, max_seq_len)
        filepaths: list[str]
    """
    labels_path = os.path.join(cache_dir, "labels.txt")
    entries = []
    with open(labels_path) as f:
        for line in f:
            idx, clase, filepath = line.strip().split("\t")
            entries.append((int(idx), int(clase), filepath))

    entries.sort(key=lambda x: x[0])
    n = len(entries)
    feature_dim = 4

    X = np.zeros((n, max_seq_len, feature_dim), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    mask = np.zeros((n, max_seq_len), dtype=np.float32)
    filepaths = []

    n_truncated = 0
    for i, (idx, clase, filepath) in enumerate(entries):
        feats = np.load(os.path.join(cache_dir, f"trial_{idx:04d}.npy"))  # (T, 4)
        T = feats.shape[0]

        if T > max_seq_len:
            # Truncar: tomar el segmento central (evita sesgar hacia el
            # arranque/frenado del trial, donde la marcha es menos representativa)
            start = (T - max_seq_len) // 2
            feats = feats[start:start + max_seq_len]
            T = max_seq_len
            n_truncated += 1

        X[i, :T, :] = feats
        mask[i, :T] = 1.0
        y[i] = clase
        filepaths.append(filepath)

    if n_truncated > 0:
        print(f"  Aviso: {n_truncated}/{n} trials truncados a max_seq_len={max_seq_len} "
              f"(se tomó el segmento central).")

    return X, y, mask, filepaths


class FerrariFeatureDataset(Dataset):
    def __init__(self, X, y, mask):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.mask = torch.as_tensor(mask, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {"x": self.X[idx], "y": self.y[idx], "mask": self.mask[idx]}


def split_dataset(X, y, mask, val_split, test_split, seed):
    """División train/val/test, estratificada de forma simple por permutación."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)

    n_test = int(n * test_split)
    n_val = int(n * val_split)

    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    def subset(indices):
        return X[indices], y[indices], mask[indices]

    return subset(train_idx), subset(val_idx), subset(test_idx)


if __name__ == "__main__":
    X, y, mask, filepaths = load_cached_dataset(
        CFG.paths.features_cache_dir, CFG.data.max_seq_len_features
    )
    print(f"X: {X.shape}  y: {y.shape}  mask: {mask.shape}")
    print(f"Distribución de clases: {np.bincount(y)}")
    print(f"Longitud real promedio (antes de padding): {mask.sum(axis=1).mean():.1f} frames")
    print(f"% de trials que llenan completamente max_seq_len (posible truncado): "
          f"{(mask.sum(axis=1) == CFG.data.max_seq_len_features).mean()*100:.1f}%")
