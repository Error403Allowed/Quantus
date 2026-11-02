import torch.nn as nn


class StockClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden1: int = 32,
        hidden2: int = 16,
        output_dim: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden1, hidden2)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden2, output_dim)

    def forward(self, x):
        # x: (batch, lookback, input_dim)
        _, hidden = self.gru(x)
        # hidden: (1, batch, hidden1) - drop the layer dim, keep batch/hidden
        h = hidden.squeeze(0)
        h = self.dropout(self.relu(self.fc(h)))
        return self.output(h)
