from scipy.stats import spearmanr

from config.schema import load_config
from trainers.train_alpha import load_bundle

config = load_config()
merged = load_bundle(config)

# each feature alone, most recent day of every test window, against the real 5-day return
for j, name in enumerate(merged.feature_columns):
    ic, _ = spearmanr(merged.X_test[:, -1, j], merged.r_test)
    print(f"{name:16s} IC {ic:+.4f}")
    