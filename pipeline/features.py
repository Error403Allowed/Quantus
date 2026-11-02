import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


# only looks backwards, so it's safe for training and for live prediction
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["Return_1"] = out["Close"].pct_change(1)
    out["Return_5"] = out["Close"].pct_change(5)
    out["Return_10"] = out["Close"].pct_change(10)

    sma10 = out["Close"].rolling(10).mean()
    sma20 = out["Close"].rolling(20).mean()
    out["Price_vs_SMA10"] = out["Close"] / (sma10 + 1e-8) - 1
    out["Price_vs_SMA20"] = out["Close"] / (sma20 + 1e-8) - 1

    out["Volatility_10"] = out["Return_1"].rolling(10).std()

    vol_mean = out["Volume"].rolling(20).mean()
    vol_std = out["Volume"].rolling(20).std()
    out["Volume_Z"] = (out["Volume"] - vol_mean) / (vol_std + 1e-8)

    out["RSI_14"] = compute_rsi(out["Close"], window=14)

    out["Vol_20"] = out["Return_1"].rolling(20).std()

    return out.dropna()


# training only: needs future prices, so never call this at inference
def add_labels(df, horizon, k):
    out = df.copy()
    out["Future_Return"] = out["Close"].shift(-horizon) / out["Close"] - 1
    out = out.dropna(subset=["Future_Return"])

    # band widens in volatile periods and for volatile stocks
    band = k * out["Vol_20"] * np.sqrt(horizon)
    out["Target"] = np.select(
        [out["Future_Return"] > band, out["Future_Return"] < -band],
        [1, -1],
        default=0,
    ).astype(np.int64)
    return out
