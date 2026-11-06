import pandas as pd

from pipeline.price_fetcher import fetch_price_data
from models.alpha.dataset import prepare_dataset, build_feature_columns, build_fold_data
from models.alpha.folds import build_trading_calendar, generate_walk_forward_folds
from config.schema import load_config

config = load_config()

# just testing with a few tickers for now, no need to pull the full list
tickers = config.data.tickers[:3]
prepared_by_ticker = {}
for ticker in tickers:
    raw = fetch_price_data(ticker, period=config.data.period, interval=config.data.interval)
    prepared_by_ticker[ticker] = prepare_dataset(raw, horizon=config.features.horizon)

calendar = build_trading_calendar(list(prepared_by_ticker.values()))
folds = list(generate_walk_forward_folds(calendar))

feature_cols = build_feature_columns(
    next(iter(prepared_by_ticker.values())),
    config.features.drop_columns,
)

# just check the first fold for now, that's enough to tell if the mechanics work
fold_data = build_fold_data(folds[0], prepared_by_ticker, feature_cols)

print(f"Fold {fold_data.fold.fold_id}: {fold_data.fold.train_start.date()} -> {fold_data.fold.train_end.date()}")
print("Tickers present in this fold's train set:", list(fold_data.train_by_ticker.keys()))

pooled_train = pd.concat([df[feature_cols] for df in fold_data.train_by_ticker.values()])

# these should land close to 0 and 1 - that's the scaler doing its job
print("\nPooled mean (should be ~0):")
print(pooled_train.mean().round(3))
print("\nPooled std (should be ~1):")
print(pooled_train.std().round(3))

# these should NOT all be ~0 - if they are, something's wrong and it's
# only actually seeing one ticker's data per fold
print("\nPer-ticker means (expected to vary):")
for ticker, df in fold_data.train_by_ticker.items():
    print(ticker, df[feature_cols].mean().round(2).to_dict())
    