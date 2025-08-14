import torch
import torch.nn as nn

class TimeLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout if num_layers>1 else 0.0,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.out_dim = hidden_size * (2 if bidirectional else 1)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, (h_n, c_n) = self.lstm(x)
        # take last timestep output
        last = out[:, -1, :]  # (batch, out_dim)
        return last
