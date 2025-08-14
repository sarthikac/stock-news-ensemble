# src/models/ensemble.py
import torch
import torch.nn as nn
from .time_lstm import TimeLSTM
from .text_encoder import TextEncoder

class EnsembleModel(nn.Module):
    def __init__(self, n_features, lstm_hidden=64, text_model_name="distilbert-base-uncased", combine_hidden=128, freeze_text=False):
        super().__init__()
        self.time_net = TimeLSTM(n_features, hidden_size=lstm_hidden)
        self.text_net = TextEncoder(pretrained_model_name=text_model_name, freeze_base=freeze_text)
        combined_dim = self.time_net.out_dim + self.text_net.out_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, combine_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(combine_hidden, 2)  # binary classification (up/down)
        )

    def forward(self, tech_seq, input_ids, attention_mask):
        # tech_seq: (batch, seq_len, n_features)
        t_emb = self.time_net(tech_seq)
        txt_emb = self.text_net(input_ids, attention_mask)
        combined = torch.cat([t_emb, txt_emb], dim=1)
        logits = self.classifier(combined)
        return logits
