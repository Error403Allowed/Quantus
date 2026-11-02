import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.schema import load_config
from models.alpha.dataset import make_inference_window, make_windows
from pipeline.features import add_labels, build_features


def fake_prices(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2022-01-03", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": rng.integers(1_000_000, 2_000_000, n),
        },
        index=idx,
    )


# features at shared dates must not change when future rows are removed
def test_no_lookahead():
    df = fake_prices()
    full = build_features(df)
    cut = build_features(df.iloc[:-5])
    np.testing.assert_allclose(full.loc[cut.index].to_numpy(), cut.to_numpy())


# the training window and the serving window for the same date must be identical
def test_train_serve_parity():
    cfg = load_config()
    cols = cfg.features.columns
    lookback = cfg.features.lookback

    feats = build_features(fake_prices())
    labeled = add_labels(feats, cfg.features.horizon, cfg.features.label_k)
    scaler = StandardScaler().fit(feats[cols])

    scaled = labeled.copy()
    scaled[cols] = scaler.transform(scaled[cols])
    X, _ = make_windows(scaled, cols, lookback)
    train_side = X[-1]

    cutoff = labeled.index[-1]
    serve_side = make_inference_window(feats.loc[:cutoff], cols, lookback, scaler)[0]

    np.testing.assert_allclose(train_side, serve_side, atol=1e-5)
