import numpy as np
import pandas as pd

from pipeline.indicators import compute_indicators

horizon = 20
buy_threshold = 0.03
sell_threshold = -0.03


def prepare_dataset(price_data: pd.DataFrame) -> pd.DataFrame:
    if price_data is None or price_data.empty:
        raise ValueError("Price data is empty or None")

    processed = compute_indicators(price_data)
    future_close = processed["Close"].shift(-horizon)
    processed["Future_Return"] = (future_close - processed["Close"]) / processed["Close"]
    processed = processed.dropna(subset=["Future_Return"])

    processed["Target"] = np.where(
        processed["Future_Return"] >= buy_threshold, 2,
        np.where(processed["Future_Return"] <= sell_threshold, 0, 1)
    )


    return processed
