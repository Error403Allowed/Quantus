import torch

from models.alpha.model import StockClassifier

batch_size = 4
lookback = 10
num_features = 8  # whatever final feature count is after the SMA drop

model = StockClassifier(input_dim=num_features)
dummy_x = torch.randn(batch_size, lookback, num_features)

out = model(dummy_x)
print("output shape:", out.shape)  # expect (4, 3)
