import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
import numpy as np

# Compute technical indicators for the given price data
def compute_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    close = data["Close"].squeeze()
    volume = data["Volume"].squeeze()

    if not isinstance(close, pd.Series) or not isinstance(volume, pd.Series):
        raise ValueError("Close and Volume must be pandas Series")


    # Compute Indicators
    data["RSI"] = RSIIndicator(close=close, window=14).rsi()
    data["MACD"] = MACD(close=close, window_slow=26, window_fast=12, window_sign=9).macd()
    data["Return_5d"] = close.pct_change(5)
    data["High_20d_dist"] = (close - close.rolling(20).max()) / close.rolling(20).max()
    data["Vol_ratio"] = volume / volume.rolling(20).mean()
    data["Volatility"] = close.rolling(window=20).std() / close  # normalize by price

    ema20 = EMAIndicator(close=close, window=20).ema_indicator()
    ema50 = EMAIndicator(close=close, window=50).ema_indicator()
    ema200 = EMAIndicator(close=close, window=200).ema_indicator()

    data["Price_to_EMA20"]  = close / ema20 - 1
    data["Price_to_EMA50"]  = close / ema50 - 1
    data["Price_to_EMA200"] = close / ema200 - 1
    data["EMA20_to_EMA50"]  = ema20 / ema50 - 1
    data["EMA50_to_EMA200"] = ema50 / ema200 - 1

    data.dropna(inplace=True)

    if data.empty: 
        raise ValueError("Not enough data to compute indicators")

    return data

# Get the latest row of indicators for prediction
def get_latest_data(data: pd.DataFrame) -> pd.Series:
    if data.empty:
        raise ValueError("DataFrame is empty")
    return (data.iloc[-1].drop(["Open", "High", "Low", "Close", "Volume"], errors="ignore")
            .fillna(0)
            .astype(float))
