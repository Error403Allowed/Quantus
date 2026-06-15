import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
import numpy as np

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
    data["EMA_20"] = EMAIndicator(close=close, window=20).ema_indicator()
    data["EMA_50"] = EMAIndicator(close=close, window=50).ema_indicator()
    data["EMA_200"] = EMAIndicator(close=close, window=200).ema_indicator()
    data["Price_Change"] = close.pct_change()
    data["Vol_Change"] = volume.pct_change()
    data["Volatility"] = close.rolling(window=20).std()
    data["MA_20"] = close.rolling(window=20).mean()
    data["MA_50"] = close.rolling(window=50).mean()
    data["MA_200"] = close.rolling(window=200).mean()
    
    data.dropna(inplace=True)

    if data.empty: 
        raise ValueError("Not enough data to compute indicators")

    return data
