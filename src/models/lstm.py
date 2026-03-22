import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim,
            batch_first=True,
            dropout=dropout if layer_dim > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hx=None):
        out, (hn, cn) = self.lstm(x, hx)
        out = self.fc(out[:, -1, :])
        return out, hn, cn