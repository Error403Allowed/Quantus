import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score

from config.schema import load_config
from models.alpha.calibration import fit_temperature, save_temperature
from models.alpha.dataset import (
    build_alpha_bundle,
    save_metadata,
    save_scaler,
    to_tensor_dataloader,
)
from models.alpha.model import StockClassifier
from pipeline.features import add_labels, build_features
from pipeline.price_fetcher import fetch_price_data


def collect_predictions(model: nn.Module, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            ps.append(torch.argmax(model(batch_X), dim=1))
            ys.append(batch_y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def report(name: str, model: nn.Module, loader) -> None:
    y_true, y_pred = collect_predictions(model, loader)
    acc = 100.0 * (y_true == y_pred).mean()
    # accuracy you'd get by always guessing the most common class in this split
    baseline = 100.0 * np.bincount(y_true, minlength=3).max() / len(y_true)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"{name}: acc {acc:.2f}% | majority baseline {baseline:.2f}% | macro-F1 {macro_f1:.3f}")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))


def collect_val_logits_and_labels(model: nn.Module, val_loader):
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            logits_list.append(model(batch_X))
            labels_list.append(batch_y)
    return torch.cat(logits_list), torch.cat(labels_list)


def train_alpha():
    torch.manual_seed(0)
    config = load_config()
    Path(config.paths.alpha_dir).mkdir(parents=True, exist_ok=True)

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

    merged = build_alpha_bundle(frames, config)
    save_scaler(merged.scaler, config.paths.scaler_path)

    train_loader = to_tensor_dataloader(
        merged.X_train, merged.y_train, config.training.batch_size, shuffle=True
    )
    val_loader = to_tensor_dataloader(
        merged.X_val, merged.y_val, config.training.batch_size, shuffle=False
    )
    test_loader = to_tensor_dataloader(
        merged.X_test, merged.y_test, config.training.batch_size, shuffle=False
    )

    input_dim = merged.X_train.shape[2]  # number of features per time step
    model = StockClassifier(
        input_dim=input_dim,
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
        for batch_X, batch_y in train_loader:
            optimiser.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimiser.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                total_val_loss += criterion(model(batch_X), batch_y).item()

        y_true, y_pred = collect_predictions(model, val_loader)
        val_f1 = f1_score(y_true, y_pred, average="macro")

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        # every 5 epochs, since patience is now short and training stops early
        if epoch % 5 == 0:
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
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    report("Train", model, train_loader)
    report("Val", model, val_loader)
    report("Test", model, test_loader)

    torch.save(model.state_dict(), config.paths.model_path)
    save_metadata(
        config.paths.metadata_path,
        feature_columns=merged.feature_columns,
        lookback=config.features.lookback,
        horizon=config.features.horizon,
        label_k=config.features.label_k,
        input_dim=input_dim,
    )
    print(f"Saved alpha model to {config.paths.model_path}")

    val_logits, val_labels = collect_val_logits_and_labels(model, val_loader)
    temperature = fit_temperature(val_logits, val_labels)
    save_temperature(temperature, config.paths.temperature_path)
    print(f"Saved temperature {temperature:.4f} to {config.paths.temperature_path}")


if __name__ == "__main__":
    train_alpha()
