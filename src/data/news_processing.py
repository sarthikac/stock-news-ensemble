import pandas as pd
from transformers import AutoTokenizer
import torch

TOKENIZER_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

def load_news_csv(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=['headline'])
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df

def aggregate_headlines_by_date(news_df: pd.DataFrame):
    """Aggregate all headlines per day into a single text (simple baseline)."""
    news_df['date'] = pd.to_datetime(news_df['date']).dt.date
    grouped = news_df.groupby('date')['headline'].apply(lambda lst: " . ".join(lst)).reset_index()
    grouped.rename(columns={'headline': 'headlines_text'}, inplace=True)
    return grouped

def encode_headlines(texts, tokenizer=tokenizer, max_length=128, device="cpu"):
    """Return tokenized inputs suitable for feeding to HF model.
    We'll return tensors for input_ids and attention_mask.
    """
    enc = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    return enc.to(device)
