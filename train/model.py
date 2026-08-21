import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import copy
from train.dataset import horizon

from train.dataset import prepare_dataset
from pipeline.data_fetcher import fetch_price_data

# 1. Define the model
class StockClassifier(nn.Module):
    def __init__(self, input_dim=11, hidden1=64, hidden2=32, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden2, output_dim)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))   # 11 -> 64
        x = self.relu(self.fc2(x))   # 64 -> 32
        x = self.output(x)           # 32 -> 3
        return x

# 2. Prepare data
price_data = fetch_price_data("AAPL", period="5y", interval="1d")
processed = prepare_dataset(price_data)

X = processed.drop(columns=["Open", "High", "Low", "Close", "Volume", "Future_Return", "Target"], errors="ignore")
y = processed["Target"]

# Convert to numpy and -1,0,1 -> 0,1,2 for PyTorch
X_np = X.values.astype(np.float32)
y_np = (y + 1).astype(np.int64)

# Train/test/val split
n = len(X_np)
i_train, i_val = int(n * 0.6), int(n * 0.8)

X_train, y_train = X_np[: i_train - horizon], y_np[: i_train - horizon]
X_val,   y_val   = X_np[i_train : i_val - horizon], y_np[i_train : i_val - horizon]
X_test,  y_test  = X_np[i_val :], y_np[i_val :]

# Convert to tensors directly from NumPy arrays
X_train_t = torch.from_numpy(X_train).float()
y_train_t = torch.from_numpy(y_train.values).long()
X_test_t = torch.from_numpy(X_test).float()
y_test_t = torch.from_numpy(y_test.values).long()
X_val_t = torch.from_numpy(X_val).float()
y_val_t = torch.from_numpy(y_val.values).long()


# Create DataLoaders with TensorDataset
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

val_dataset = TensorDataset(X_val_t, y_val_t)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

best_val_loss = float("inf")
no_improve_epochs = 0
patience = 75
best_state = None
num_epochs = 1000

# 3. Create model, loss, optimizer
model = StockClassifier()
criterion = nn.CrossEntropyLoss()  # for 3-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            val_loss += criterion(model(batch_X), batch_y).item()
    val_loss /= len(val_loader)

    if epoch % 25 == 0:
        print(f"Epoch {epoch}, Train loss: {total_loss/len(train_loader):.4f}, Val loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_epochs = 0
        best_state = copy.deepcopy(model.state_dict())
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

if best_state is not None:
    model.load_state_dict(best_state)

# Pre-model evaluation
model.eval()
train_correct, train_total = 0, 0
with torch.no_grad():
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == batch_y).sum().item()
        train_total += batch_y.size(0)
train_accuracy = 100 * train_correct / train_total

test_correct, total = 0, 0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)
test_accuracy = 100 * test_correct / total

print(f"Train Accuracy: {train_accuracy:.2f}%")
print(f"Test Accuracy: {test_accuracy:.2f}%")

torch.save(model.state_dict(), 'models/stock_classifier.pt')
print('Model saved to models/stock_classifier.pt')

# 6. Temperature scaling: find T that minimizes NLL on the test set
logits_list, labels_list = [], []
with torch.no_grad():
    for batch_X, batch_y in val_loader:
        logits_list.append(model(batch_X))
        labels_list.append(batch_y)
val_logits = torch.cat(logits_list)
val_labels = torch.cat(labels_list)

temp_param = nn.Parameter(torch.tensor(1.0))
temp_opt = optim.LBFGS([temp_param], lr=0.01, max_iter=50)


def temp_nll() -> torch.Tensor:
    temp_opt.zero_grad()
    loss = nn.functional.cross_entropy(val_logits / temp_param, val_labels)
    loss.backward()
    return loss


temp_opt.step(temp_nll)
temp_value = float(temp_param.item())
torch.save({'temperature': temp_value}, 'models/temperature.pt')
print(f'Temperature (T): {temp_value:.4f}')

