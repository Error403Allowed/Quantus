import json
from dataclasses import dataclass
from typing import Any, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config.schema import QuantusConfig

@dataclass
class AlphaDataBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    r_train: np.ndarray
    d_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    r_val: np.ndarray
    d_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    r_test: np.ndarray
    d_test: np.ndarray
    feature_columns: List[str]
    scaler: StandardScaler
    feature_columns: List[str]
    scaler: StandardScaler
    regime_scaler: Any = None
    regime_gmm: Any = None


def make_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feats = df[feature_cols].to_numpy(dtype=np.float32)
    labels = df["Target"].to_numpy(dtype=np.int64) + 1  # -1/0/1 -> 0/1/2
    rets = df["Future_Return"].to_numpy(dtype=np.float64)
    dates = df.index.to_numpy()

    X, y, r, d = [], [], [], []
    # window ends AT row i (inclusive), same as tail(lookback) at inference
    for i in range(lookback - 1, len(df)):
        X.append(feats[i - lookback + 1 : i + 1])  # keep (lookback, n_features)
        y.append(labels[i])
        r.append(rets[i])
        d.append(dates[i])

    if X:
        X_arr = np.stack(X).astype(np.float32)
    else:
        X_arr = np.empty((0, lookback, len(feature_cols)), dtype=np.float32)
    return (
        X_arr,
        np.array(y, dtype=np.int64),
        np.array(r, dtype=np.float64),
        np.array(d, dtype="datetime64[ns]"),
    )


def split_by_time(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    horizon: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # the last `horizon` rows of each split have labels that peek into the next split
    train_df = df.iloc[: train_end - horizon].copy()
    val_df = df.iloc[train_end : val_end - horizon].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def build_alpha_bundle(
    frames: List[pd.DataFrame],
    config: QuantusConfig,
) -> AlphaDataBundle:
    cols = config.features.columns
    lookback = config.features.lookback

    # 1) split every ticker by time, nothing scaled yet
    splits = [
        split_by_time(
            f,
            config.split.train_ratio,
            config.split.val_ratio,
            config.features.horizon,
        )
        for f in frames
    ]

    # 2) one scaler, fit only on the pooled training rows
    all_train = pd.concat([s[0] for s in splits])
    scaler = StandardScaler().fit(all_train[cols])

    # 3) scale each piece, window it per ticker, then stack
    X = {"train": [], "val": [], "test": []}
    y = {"train": [], "val": [], "test": []}
    r = {"train": [], "val": [], "test": []}
    d = {"train": [], "val": [], "test": []}
    for parts in splits:
        for name, part in zip(("train", "val", "test"), parts):
            part = part.copy()
            part[cols] = scaler.transform(part[cols])
            xi, yi, ri, di = make_windows(part, cols, lookback)
            X[name].append(xi)
            y[name].append(yi)
            r[name].append(ri)
            d[name].append(di)

    return AlphaDataBundle(
        X_train=np.concatenate(X["train"]),
        y_train=np.concatenate(y["train"]),
        r_train=np.concatenate(r["train"]),
        d_train=np.concatenate(d["train"]),
        X_val=np.concatenate(X["val"]),
        y_val=np.concatenate(y["val"]),
        r_val=np.concatenate(r["val"]),
        d_val=np.concatenate(d["val"]),
        X_test=np.concatenate(X["test"]),
        y_test=np.concatenate(y["test"]),
        r_test=np.concatenate(r["test"]),
        d_test=np.concatenate(d["test"]),
        feature_columns=cols,
        scaler=scaler,
    )


# serving-side twin of make_windows: one window, latest rows, same scaler
def make_inference_window(
    df: pd.DataFrame,
    feature_cols: List[str],
    lookback: int,
    scaler: StandardScaler,
) -> np.ndarray:
    if len(df) < lookback:
        raise ValueError(f"Need at least {lookback} rows of features, got {len(df)}")

    window = df[feature_cols].tail(lookback).copy()
    window[feature_cols] = scaler.transform(window[feature_cols])
    # (1, lookback, n_features): a batch of one window
    return window.to_numpy(dtype=np.float32)[None, :, :]

def to_tensor_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).long(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def save_scaler(scaler: StandardScaler, path: str) -> None:
    joblib.dump(scaler, path)


def load_scaler(path: str) -> StandardScaler:
    return joblib.load(path)


def save_metadata(path: str, **meta) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_metadata(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    