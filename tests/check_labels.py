import numpy as np

from config.schema import load_config
from models.alpha.dataset import build_alpha_bundle
from pipeline.features import add_labels, build_features
from pipeline.price_fetcher import fetch_price_data

config = load_config()

# download and build features once, labels get rebuilt for each k
feats = {
    t: build_features(fetch_price_data(t, config.data.period, config.data.interval))
    for t in config.data.tickers
}

for k in [0.3, 0.5, 0.7, 1.0]:
    frames = [add_labels(f, config.features.horizon, k) for f in feats.values()]
    bundle = build_alpha_bundle(frames, config)
    share = np.bincount(bundle.y_train, minlength=3) / len(bundle.y_train)
    print(f"k={k}: SELL {share[0]:.2f} HOLD {share[1]:.2f} BUY {share[2]:.2f}")
    