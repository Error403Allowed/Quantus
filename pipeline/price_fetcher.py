from datetime import datetime, timedelta, timezone

import finnhub
import pandas as pd
import yfinance as yf

_REQUIRED_COLS = {"Open", "High", "Low", "Close", "Volume"}


def fetch_price_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

    if data is None or data.empty:
        raise ValueError(f"Could not fetch price data for {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    missing = _REQUIRED_COLS - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns for {ticker}: {missing}")

    return data
