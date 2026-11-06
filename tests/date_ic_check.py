import numpy as np
import pandas as pd

from config.schema import load_config
from trainers.train_alpha import collect_probs, fit_model, load_bundle

config = load_config()
merged = load_bundle(config)
model, loaders = fit_model(config, merged, seed=0, log_every=0)

# the test loader isn't shuffled, so rows line up with merged.*_test
_, probs = collect_probs(model, loaders["test"])
vol_idx = merged.feature_columns.index("Volume_Z")

df = pd.DataFrame({
    "date": merged.d_test,
    "ret": merged.r_test,
    "model": probs[:, 2] - probs[:, 0],
    "volume_z": merged.X_test[:, -1, vol_idx],
})


def per_date_ic(col: str) -> None:
    # one rank correlation per date: how well does the score order that day's stocks?
    ics = pd.Series(
        {date: g[col].corr(g["ret"], method="spearman") for date, g in df.groupby("date")}
    ).dropna()
    icir = ics.mean() / ics.std()
    print(
        f"{col:9s} per-date IC {ics.mean():+.4f} | ICIR {icir:+.3f} | "
        f"t-stat {icir * np.sqrt(len(ics)):+.2f} | {len(ics)} dates"
    )


for col in ("model", "volume_z"):
    print(f"{col:9s} pooled IC   {df[col].corr(df['ret'], method='spearman'):+.4f}")
per_date_ic("model")
per_date_ic("volume_z") 
