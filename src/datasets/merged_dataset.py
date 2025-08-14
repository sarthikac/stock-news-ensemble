# src/datasets/merged_dataset.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime
from ..data.technical_indicators import add_technical_indicators
from ..data.news_processing import load_news_csv, aggregate_headlines_by_date

class MergedDataset(Dataset):
    """
    Prepares sequences of technical indicator vectors + corresponding day's headlines text.
    Predict next-day direction: 1 if Close_next > Close_today else 0
    """
    def __init__(self, price_csv, news_csv=None, seq_len=30, transform=None, device="cpu"):
        self.device = device
        prices = pd.read_csv(price_csv, parse_dates=["date"])
        prices = add_technical_indicators(prices)
        prices['date'] = pd.to_datetime(prices['date']).dt.date

        self.seq_len = seq_len

        if news_csv and os.path.exists(news_csv):
            news = load_news_csv(news_csv)
            news_grouped = aggregate_headlines_by_date(news)
            news_grouped['date'] = pd.to_datetime(news_grouped['date']).dt.date
            merged = pd.merge(prices, news_grouped, on='date', how='left')
            merged['headlines_text'].fillna('', inplace=True)
        else:
            prices['headlines_text'] = ''
            merged = prices

        merged.sort_values('date', inplace=True)
        merged.reset_index(drop=True, inplace=True)

        # Build label = whether next day close increases
        merged['close_next'] = merged['Close'].shift(-1)
        merged.dropna(inplace=True)
        merged['label'] = (merged['close_next'] > merged['Close']).astype(int)

        # Feature columns (pick numeric indicator columns)
        ignore_cols = {'date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'headlines_text', 'close_next', 'label'}
        feature_cols = [c for c in merged.columns if c not in ignore_cols and merged[c].dtype in [float, int]]
        self.feature_cols = feature_cols

        self.data = merged

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # sequence of technical features ending at idx+seq_len-1, label at idx+seq_len-1
        start = idx
        end = idx + self.seq_len
        seq = self.data.iloc[start:end]
        tech = seq[self.feature_cols].values.astype(np.float32)  # shape (seq_len, n_features)
        # text is the last day's headlines_text
        text = seq['headlines_text'].iloc[-1]
        label = int(self.data['label'].iloc[end - 1])
        sample = {
            'tech': torch.tensor(tech, dtype=torch.float32).to(self.device),
            'text': text,
            'label': torch.tensor(label, dtype=torch.long).to(self.device),
            'date': str(self.data['date'].iloc[end - 1])
        }
        return sample
