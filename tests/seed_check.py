import numpy as np

from config.schema import load_config
from trainers.train_alpha import fit_model, load_bundle, report

config = load_config()
merged = load_bundle(config)
prior = np.bincount(merged.y_train, minlength=3) / len(merged.y_train)

rows = []
for seed in range(5):
    model, loaders = fit_model(config, merged, seed, log_every=0)
    m = report("Test", model, loaders["test"], merged.r_test, prior, verbose=False)
    m["loss_gain"] = m["prior_loss"] - m["loss"]  # positive = beats the class-share model
    m["gap"] = m["acc"] - m["baseline"]
    rows.append(m)
    print(f"seed {seed}: acc gap {m['gap']:+.2f} | loss gain {m['loss_gain']:+.4f} | IC {m['ic']:+.4f}")

print()
for key in ("gap", "loss_gain", "ic"):
    vals = np.array([m[key] for m in rows])
    print(f"{key}: mean {vals.mean():+.4f} | std {vals.std(ddof=1):.4f}")