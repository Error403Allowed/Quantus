import copy
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from pipeline.price_fetcher import fetch_price_data
from train.dataset import horizon, prepare_dataset

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "JNJ"]
LOOKBACK = 10
DROP_COLS = {"Open", "High", "Low", "Close", "Volume", "Future_Return", "Target"}
BATCH_SIZE = 64
NUM_EPOCHS = 300
PATIENCE = 30
LR = 5e-4
WEIGHT_DECAY = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StockClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden1: int = 32, hidden2: int = 16, output_dim: int = 3, p: float = 0.45):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_windows(df: "pd.DataFrame", feature_cols: list, lookback: int):
    feats = df[feature_cols].values.astype(np.float32)
    labels = (df["Target"].to_numpy(dtype=np.int64)) # -1,0,1 -> 0,1,2
    X, y = [], []
    for i in range(lookback, len(df)):
        X.append(feats[i - lookback:i].reshape(-1))
        y.append(labels[i])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def build_splits():
    per_ticker: list = []
    all_train_frames: list = []
    feature_cols: list[str] = []  # <-- fixes Pylance reportPossiblyUnbound

    for ticker in TICKERS:
        df = prepare_dataset(fetch_price_data(ticker, period="5y", interval="1d"))
        feature_cols = [c for c in df.columns if c not in DROP_COLS]

        m = len(df)
        a, b = int(m * 0.6), int(m * 0.8)

        tr_df = df.iloc[: a - horizon].copy()
        va_df = df.iloc[a: b - horizon].copy()
        te_df = df.iloc[b:].copy()

        if any(len(split) <= LOOKBACK for split in (tr_df, va_df, te_df)):
            raise ValueError(f"Not enough rows after split for {ticker}")

        all_train_frames.append(tr_df[feature_cols].copy())
        per_ticker.append((ticker, tr_df, va_df, te_df, feature_cols))

    # Fit ONE global scaler on all train data across all tickers
    scaler = StandardScaler()
    scaler.fit(np.vstack([f.values for f in all_train_frames]))

    Xtr, ytr, Xva, yva, Xte, yte = [], [], [], [], [], []

    for _ticker, tr_df, va_df, te_df, fcols in per_ticker:
        tr_df.loc[:, fcols] = scaler.transform(tr_df[fcols].values)
        va_df.loc[:, fcols] = scaler.transform(va_df[fcols].values)
        te_df.loc[:, fcols] = scaler.transform(te_df[fcols].values)

        Xw_tr, yw_tr = make_windows(tr_df, fcols, LOOKBACK)
        Xw_va, yw_va = make_windows(va_df, fcols, LOOKBACK)
        Xw_te, yw_te = make_windows(te_df, fcols, LOOKBACK)

        Xtr.append(Xw_tr); ytr.append(yw_tr)
        Xva.append(Xw_va); yva.append(yw_va)
        Xte.append(Xw_te); yte.append(yw_te)

    X_train = np.concatenate(Xtr)
    y_train = np.concatenate(ytr)
    X_val = np.concatenate(Xva)
    y_val = np.concatenate(yva)
    X_test = np.concatenate(Xte)
    y_test = np.concatenate(yte)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_cols


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def compute_class_weights(y_train: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=3).astype(np.float32)
    weights = counts.sum() / (3.0 * np.maximum(counts, 1.0))
    return torch.tensor(weights / weights.mean(), dtype=torch.float32, device=DEVICE)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for bX, by in loader:
            bX, by = bX.to(DEVICE), by.to(DEVICE)
            logits = model(bX)
            total_loss += criterion(logits, by).item()
            correct += (logits.argmax(dim=1) == by).sum().item()
            count += by.size(0)
    return total_loss / len(loader), 100.0 * correct / count


def fit_temperature(model: nn.Module, val_loader: DataLoader) -> float:
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for bX, by in val_loader:
            logits_list.append(model(bX.to(DEVICE)).cpu())
            labels_list.append(by)
    val_logits = torch.cat(logits_list)
    val_labels = torch.cat(labels_list)

    temperature = nn.Parameter(torch.tensor(1.0))
    opt = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = nn.functional.cross_entropy(val_logits / torch.clamp(temperature, min=1e-3), val_labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(temperature.detach(), min=1e-3).item())


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_cols = build_splits()

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)
    test_loader = make_loader(X_test, y_test, BATCH_SIZE, shuffle=False)

    print("Train:", np.bincount(y_train, minlength=3))
    print("Val:  ", np.bincount(y_val, minlength=3))
    print("Test: ", np.bincount(y_test, minlength=3))

    model = StockClassifier(input_dim=X_train.shape[1]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(y_train),
        label_smoothing=0.05,
    )
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for bX, by in train_loader:
            bX, by = bX.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bX), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d} | Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | Val acc: {val_acc:.1f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    assert best_state is not None, "No best state captured — training failed"
    model.load_state_dict(best_state)

    _, train_acc = evaluate(model, train_loader, criterion)
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Test  Accuracy: {test_acc:.2f}%")

    torch.save(model.state_dict(), MODEL_DIR / "stock_classifier.pt")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(
        {"lookback": LOOKBACK, "feature_cols": feature_cols, "input_dim": int(X_train.shape[1])},
        MODEL_DIR / "training_metadata.pkl",
    )

    temperature = fit_temperature(model, val_loader)
    torch.save({"temperature": temperature}, MODEL_DIR / "temperature.pt")

    print(f"Temperature (T): {temperature:.4f}")
    print("Model saved to models/")


if __name__ == "__main__":
    main()
