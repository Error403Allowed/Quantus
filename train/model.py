import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

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
        x = self.relu(self.fc1(x))   # 11 → 64
        x = self.relu(self.fc2(x))   # 64 → 32
        x = self.output(x)           # 32 → 3
        return x

# 2. Prepare data
price_data = fetch_price_data("AAPL", period="5y", interval="1d")
processed = prepare_dataset(price_data)

X = processed.drop(columns=["Open", "High", "Low", "Close", "Volume", "Future_Return", "Target"], errors="ignore")
y = processed["Target"]

# Convert to numpy and -1,0,1 → 0,1,2 for PyTorch
X_np = X.values
y_np = (y + 1).astype(np.int32)  # fixed: -1→0, 0→1, 1→2

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42, stratify=y_np
)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train.values.copy())
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test.values.copy())

# Create DataLoaders with TensorDataset
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. Create model, loss, optimizer
model = StockClassifier()
criterion = nn.CrossEntropyLoss()  # for 3-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    if epoch % 25 == 0:
        print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")

# 5. Evaluation
# model.eval()
# correct, total = 0, 0
# with torch.no_grad():
#     for batch_X, batch_y in test_loader:
#         outputs = model(batch_X)
#         _, predicted = torch.max(outputs, 1)
#         correct += (predicted == batch_y).sum().item()
#         total += batch_y.size(0)

# accuracy = 100 * correct / total
# print(f"Test Accuracy: {accuracy:.2f}%")

# Add this before evaluation
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

