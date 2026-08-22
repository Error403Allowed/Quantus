import pandas as pd
from pipeline.indicators import compute_indicators
from pipeline.price_fetcher import fetch_price_data
from train.dataset import prepare_dataset

df = prepare_dataset(fetch_price_data("AAPL", period="5y", interval="1d"))
drop_cols = ["Open", "High", "Low", "Close", "Volume", "Future_Return", "Target"]
feature_cols = [c for c in df.columns if c not in drop_cols]

print(df[feature_cols].corrwith(df["Future_Return"]).sort_values())

print(feature_cols)
print(len(feature_cols))
