import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# market-level features from SPY, backward-looking only
def build_regime_features(market: pd.DataFrame) -> pd.DataFrame:
    close = market["Close"]
    out = pd.DataFrame(index=market.index)
    out["Mkt_Vol_20"] = close.pct_change().rolling(20).std()
    out["Mkt_Trend"] = close / close.rolling(200).mean() - 1
    out["Mkt_Drawdown"] = close / close.rolling(252).max() - 1
    return out.dropna()


# fit on rows up to train_end only, so the regime model never sees val/test dates
def fit_regime_model(feats: pd.DataFrame, train_end, n_components: int, seed: int = 0):
    train = feats.loc[:train_end]
    scaler = StandardScaler().fit(train)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=5,
        random_state=seed,
    ).fit(scaler.transform(train))
    return scaler, gmm


# soft membership per day; columns sorted so Regime_0 is always the calmest cluster
def regime_probs(feats: pd.DataFrame, scaler, gmm) -> pd.DataFrame:
    probs = gmm.predict_proba(scaler.transform(feats))
    order = np.argsort(gmm.means_[:, 0])  # column 0 is Mkt_Vol_20 (scaled)
    probs = probs[:, order]
    cols = [f"Regime_{i}" for i in range(probs.shape[1])]
    return pd.DataFrame(probs, index=feats.index, columns=cols)


def save_regime(path: str, scaler, gmm) -> None:
    joblib.dump({"scaler": scaler, "gmm": gmm}, path)


def load_regime(path: str):
    obj = joblib.load(path)
    return obj["scaler"], obj["gmm"]
