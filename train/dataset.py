import pandas as pd
import numpy as np
from pipeline.indicators import compute_indicators

# Global variables
horizon = 20
buy_threshold = 0.03
sell_threshold = -0.03

# Prepare dataset with indicators and future returns
def prepare_dataset(price_data: pd.DataFrame) -> pd.DataFrame:
    if price_data is None or price_data.empty:
        raise ValueError("Price data is empty or None")
    
    processed = compute_indicators(price_data)
    future_close = processed["Close"].shift(-horizon)
    processed["Future_Return"] = (future_close - processed["Close"]) / processed["Close"]
    processed = processed.dropna(subset=["Future_Return"])  # Drop rows where future return cannot be computed
    
    processed["Target"] = np.where(
        processed["Future_Return"] > buy_threshold, 1,
        np.where(processed["Future_Return"] < sell_threshold, -1, 0)
    )

    return processed