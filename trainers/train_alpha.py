import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score
from scipy.stats import spearmanr

from config.schema import QuantusConfig, load_config
from models.alpha.calibration import fit_temperature, save_temperature
from models.alpha.dataset import (
    AlphaDataBundle,
    build_alpha_bundle,
    save_metadata,
    save_scaler,
    split_by_time,
    to_tensor_dataloader,
)
from models.alpha.model import StockClassifier
from pipeline.features import add_labels, build_features
from pipeline.price_fetcher import fetch_price_data
from pipeline.regime import (
    build_regime_features,
    fit_regime_model,
    regime_probs,
    save_regime,
)


def collect_probs(model: nn.Module, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            ps.append(torch.softmax(model(batch_X), dim=1))
            ys.append(batch_y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def report(name: str, model: nn.Module, loader, returns, prior, verbose: bool = True):
    y_true, probs = collect_probs(model, loader)
    y_pred = probs.argmax(axis=1)

    acc = 100.0 * (y_true == y_pred).mean()
    baseline = 100.0 * np.bincount(y_true, minlength=3).max() / len(y_true)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    # model's average cross-entropy vs a "model" that only knows the train class shares
    loss = -np.log(probs[np.arange(len(y_true)), y_true] + 1e-12).mean()
    prior_loss = -np.log(prior[y_true]).mean()

    # rank correlation between the model's lean (P(BUY) - P(SELL)) and the real 5-day return
    ic, _ = spearmanr(probs[:, 2] - probs[:, 0], returns)

    if verbose:
        print(f"{name}: acc {acc:.2f}% | baseline {baseline:.2f}% | macro-F1 {macro_f1:.3f}")
        print(f"{name}: loss {loss:.4f} | prior loss {prior_loss:.4f} | IC {ic:+.4f}")
        print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))
    return {"acc": acc, "baseline": baseline, "f1": macro_f1,
            "loss": loss, "prior_loss": prior_loss, "ic": ic}


def collect_val_logits_and_labels(model: nn.Module, val_loader):
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            logits_list.append(model(batch_X))
            labels_list.append(batch_y)
    return torch.cat(logits_list), torch.cat(labels_list)


# the slow part (downloads), so it runs once and gets reused
def load_bundle(config: QuantusConfig) -> AlphaDataBundle:
    frames = []
    for ticker in config.data.tickers:
        raw_df = fetch_price_data(
            ticker,
            period=config.data.period,
            interval=config.data.interval,
        )
        feats = build_features(raw_df)
        frames.append(
            add_labels(feats, config.features.horizon, config.features.label_k)
        )

    # regime model: fit on market data up to the last training date only
    spy = fetch_price_data(config.regime.ticker, period=config.regime.period, interval="1d")
    market = build_regime_features(spy)
    train_end = min(
        split_by_time(
            f, config.split.train_ratio, config.split.val_ratio, config.features.horizon
        )[0].index[-1]
        for f in frames
    )
    regime_scaler, regime_gmm = fit_regime_model(
        market, train_end, config.regime.n_components
    )
    probs = regime_probs(market, regime_scaler, regime_gmm)
    print("regime share of days:", probs.mean().round(2).to_dict())

    frames = [f.join(probs, how="inner") for f in frames]
    merged = build_alpha_bundle(frames, config)
    merged.regime_scaler = regime_scaler
    merged.regime_gmm = regime_gmm
    return merged


# trains one model from a fixed seed; log_every=0 turns the epoch printing off
def fit_model(config: QuantusConfig, merged: AlphaDataBundle, seed: int, log_every: int = 5):
    torch.manual_seed(seed)

    batch_size = config.training.batch_size
    loaders = {
        "train": to_tensor_dataloader(merged.X_train, merged.y_train, batch_size, shuffle=True),
        "val": to_tensor_dataloader(merged.X_val, merged.y_val, batch_size, shuffle=False),
        "test": to_tensor_dataloader(merged.X_test, merged.y_test, batch_size, shuffle=False),
    }

    model = StockClassifier(
        input_dim=merged.X_train.shape[2],  # number of features per time step
        hidden1=config.alpha.hidden1,
        hidden2=config.alpha.hidden2,
        output_dim=config.alpha.output_dim,
        dropout=config.alpha.dropout,
    )

    criterion = nn.CrossEntropyLoss()
    optimiser = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    best_val_loss = float("inf")
    no_improve_epochs = 0
    best_state = None

    for epoch in range(config.training.epochs):
        model.train()
        total_train_loss = 0.0
        for batch_X, batch_y in loaders["train"]:
            optimiser.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimiser.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in loaders["val"]:
                total_val_loss += criterion(model(batch_X), batch_y).item()

        avg_train_loss = total_train_loss / len(loaders["train"])
        avg_val_loss = total_val_loss / len(loaders["val"])

        if log_every and epoch % log_every == 0:
            y_true, probs = collect_probs(model, loaders["val"])
            val_f1 = f1_score(y_true, probs.argmax(axis=1), average="macro")
            print(
                f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= config.training.patience:
                if log_every:
                    print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, loaders


def train_alpha():
    config = load_config()
    Path(config.paths.alpha_dir).mkdir(parents=True, exist_ok=True)

    merged = load_bundle(config)
    save_scaler(merged.scaler, config.paths.scaler_path)
    save_regime(config.paths.regime_path, merged.regime_scaler, merged.regime_gmm)

    model, loaders = fit_model(config, merged, seed=0)

    prior = np.bincount(merged.y_train, minlength=3) / len(merged.y_train)
    report("Train", model, loaders["train"], merged.r_train, prior)
    report("Val", model, loaders["val"], merged.r_val, prior)
    report("Test", model, loaders["test"], merged.r_test, prior)

    torch.save(model.state_dict(), config.paths.model_path)
    save_metadata(
        config.paths.metadata_path,
        feature_columns=merged.feature_columns,
        lookback=config.features.lookback,
        horizon=config.features.horizon,
        label_k=config.features.label_k,
        input_dim=merged.X_train.shape[2],
    )
    print(f"Saved alpha model to {config.paths.model_path}")

    val_logits, val_labels = collect_val_logits_and_labels(model, loaders["val"])
    temperature = fit_temperature(val_logits, val_labels)
    save_temperature(temperature, config.paths.temperature_path)
    print(f"Saved temperature {temperature:.4f} to {config.paths.temperature_path}")


if __name__ == "__main__":
    train_alpha()