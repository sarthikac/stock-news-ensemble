import torch
import torch.nn as nn
from transformers import AutoModel

ENCODER_MODEL = "distilbert-base-uncased"

class TextEncoder(nn.Module):
    def __init__(self, pretrained_model_name=ENCODER_MODEL, freeze_base=False):
        super().__init__()
        self.base = AutoModel.from_pretrained(pretrained_model_name)
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False
        # DistilBERT returns last_hidden_state; pool by [0] token approximation: mean pooling
        self.out_dim = self.base.config.hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        last = outputs.last_hidden_state  # (batch, seq_len, hidden)
        # mean pooling (masked)
        mask = attention_mask.unsqueeze(-1)
        summed = (last * mask).sum(1)
        lengths = mask.sum(1).clamp(min=1e-9)
        pooled = summed / lengths
        return pooled  # (batch, hidden)
