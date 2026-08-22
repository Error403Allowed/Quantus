from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from config.schema import load_config
from models.alpha.calibration import fit_temperature, save_temperature
from models.alpha.dataset import (
    build_alpha_bundle_from_dataframe,
    merge_alpha_bundles,
    save_scaler,
    to_tensor_dataloader,
)
from models.alpha.model import StockClassifier
from pipeline.price_fetcher import fetch_price_data
from models.alpha.dataset import prepare_dataset


def evaluate_accuracy(model: nn.Module, loader) -> float:
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch_X, batch_y in loader:
            logits = model(batch_X)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    return 100.0 * correct / total if total > 0 else 0.0


def collect_val_logits_and_labels(model: nn.Module, val_loader):
    model.eval()
    logits_list, labels_list = [], []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            logits_list.append(model(batch_X))
            labels_list.append(batch_y)

    return torch.cat(logits_list), torch.cat(labels_list)


def train_alpha():
    config = load_config()
    Path(config.paths.alpha_dir).mkdir(parents=True, exist_ok=True)

    bundles = []
    for ticker in config.data.tickers:
        raw_df = fetch_price_data(
            ticker,
            period=config.data.period,
            interval=config.data.interval,
        )
        prepared_df = prepare_dataset(raw_df)
        bundle = build_alpha_bundle_from_dataframe(prepared_df, config)
        bundles.append(bundle)

    merged = merge_alpha_bundles(bundles)

    save_scaler(merged.scaler, config.paths.scaler_path)

    train_loader = to_tensor_dataloader(
        merged.X_train,
        merged.y_train,
        batch_size=config.training.batch_size,
        shuffle=True,
    )
    val_loader = to_tensor_dataloader(
        merged.X_val,
        merged.y_val,
        batch_size=config.training.batch_size,
        shuffle=False,
    )
    test_loader = to_tensor_dataloader(
        merged.X_test,
        merged.y_test,
        batch_size=config.training.batch_size,
        shuffle=False,
    )

    model = StockClassifier(
        input_dim=merged.X_train.shape[1],
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
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimiser.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                logits = model(batch_X)
                total_val_loss += criterion(logits, batch_y).item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        if epoch % 25 == 0:
            print(
                f"Epoch {epoch} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f}"
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

    train_acc = evaluate_accuracy(model, train_loader)
    test_acc = evaluate_accuracy(model, test_loader)

    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")

    torch.save(model.state_dict(), config.paths.model_path)
    print(f"Saved alpha model to {config.paths.model_path}")

    val_logits, val_labels = collect_val_logits_and_labels(model, val_loader)
    temperature = fit_temperature(val_logits, val_labels)
    save_temperature(temperature, config.paths.temperature_path)

    print(f"Saved temperature {temperature:.4f} to {config.paths.temperature_path}")


if __name__ == "__main__":
    train_alpha()
