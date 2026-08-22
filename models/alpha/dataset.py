from dataclasses import dataclass
from typing import List, Tuple

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
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_columns: List[str]
    scaler: StandardScaler


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def prepare_dataset(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    out = df.copy()

    out["Return_1"] = out["Close"].pct_change(1)
    out["Return_5"] = out["Close"].pct_change(5)
    out["Return_10"] = out["Close"].pct_change(10)

    out["SMA_10"] = out["Close"].rolling(10).mean()
    out["SMA_20"] = out["Close"].rolling(20).mean()
    out["Price_vs_SMA10"] = out["Close"] / (out["SMA_10"] + 1e-8) - 1
    out["Price_vs_SMA20"] = out["Close"] / (out["SMA_20"] + 1e-8) - 1

    out["Volatility_10"] = out["Return_1"].rolling(10).std()

    vol_mean = out["Volume"].rolling(20).mean()
    vol_std = out["Volume"].rolling(20).std()
    out["Volume_Z"] = (out["Volume"] - vol_mean) / (vol_std + 1e-8)

    out["RSI_14"] = compute_rsi(out["Close"], window=14)

    out["Future_Return"] = out["Close"].shift(-horizon) / out["Close"] - 1

    threshold = 0.02
    out["Target"] = np.select(
        [
            out["Future_Return"] > threshold,
            out["Future_Return"] < -threshold,
        ],
        [
            1,
            -1,
        ],
        default=0,
    ).astype(np.int64)

    out = out.dropna().reset_index(drop=True)
    return out


def make_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray]:
    feats = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    target_array = df["Target"].to_numpy(dtype=np.int64, copy=True)
    labels = target_array + 1

    X, y = [], []
    for i in range(lookback, len(df)):
        X.append(feats[i - lookback:i].flatten())
        y.append(labels[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def split_by_time(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    horizon: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[: train_end - horizon].copy()
    val_df = df.iloc[train_end : val_end - horizon].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def build_feature_columns(df: pd.DataFrame, drop_columns: List[str]) -> List[str]:
    return [c for c in df.columns if c not in drop_columns]


def fit_and_transform_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df.loc[:, feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df.loc[:, feature_cols] = scaler.transform(val_df[feature_cols])
    test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])

    return train_df, val_df, test_df, scaler


def build_alpha_bundle_from_dataframe(
    df: pd.DataFrame,
    config: QuantusConfig,
) -> AlphaDataBundle:
    feature_cols = build_feature_columns(df, config.features.drop_columns)

    train_df, val_df, test_df = split_by_time(
        df=df,
        train_ratio=config.split.train_ratio,
        val_ratio=config.split.val_ratio,
        horizon=config.features.horizon,
    )

    train_df, val_df, test_df, scaler = fit_and_transform_splits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
    )

    X_train, y_train = make_windows(train_df, feature_cols, config.features.lookback)
    X_val, y_val = make_windows(val_df, feature_cols, config.features.lookback)
    X_test, y_test = make_windows(test_df, feature_cols, config.features.lookback)

    return AlphaDataBundle(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_columns=feature_cols,
        scaler=scaler,
    )


def merge_alpha_bundles(bundles: List[AlphaDataBundle]) -> AlphaDataBundle:
    X_train = np.concatenate([b.X_train for b in bundles], axis=0)
    y_train = np.concatenate([b.y_train for b in bundles], axis=0)
    X_val = np.concatenate([b.X_val for b in bundles], axis=0)
    y_val = np.concatenate([b.y_val for b in bundles], axis=0)
    X_test = np.concatenate([b.X_test for b in bundles], axis=0)
    y_test = np.concatenate([b.y_test for b in bundles], axis=0)

    return AlphaDataBundle(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_columns=bundles[-1].feature_columns,
        scaler=bundles[-1].scaler,
    )


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
