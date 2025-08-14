import torch
import pandas as pd
import json
from data.news_processing import encode_headlines
from models.ensemble import EnsembleModel
import numpy as np

def load_model(model_path, n_features, device="cpu"):
    model = EnsembleModel(n_features=n_features, lstm_hidden=64, text_model_name="distilbert-base-uncased", combine_hidden=128, freeze_text=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def build_inference_windows(price_csv, news_csv, seq_len=30):
    # similar to dataset but simpler to obtain test dates and features
    from data.technical_indicators import add_technical_indicators
    from data.news_processing import load_news_csv, aggregate_headlines_by_date
    prices = pd.read_csv(price_csv, parse_dates=['date'])
    prices = add_technical_indicators(prices)
    prices['date'] = pd.to_datetime(prices['date']).dt.date

    if news_csv:
        news = load_news_csv(news_csv)
        news = aggregate_headlines_by_date(news)
        news['date'] = pd.to_datetime(news['date']).dt.date
        merged = pd.merge(prices, news, on='date', how='left')
        merged['headlines_text'].fillna('', inplace=True)
    else:
        merged = prices
        merged['headlines_text'] = ''

    merged.sort_values('date', inplace=True)
    merged.reset_index(drop=True, inplace=True)
    merged['close_next'] = merged['Close'].shift(-1)
    merged.dropna(inplace=True)
    ignore_cols = {'date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'headlines_text', 'close_next'}
    feature_cols = [c for c in merged.columns if c not in ignore_cols and merged[c].dtype in [float, int]]

    windows = []
    for i in range(len(merged)-seq_len):
        seq = merged.iloc[i:i+seq_len]
        tech = seq[feature_cols].values.astype(np.float32)
        text = seq['headlines_text'].iloc[-1]
        date = merged['date'].iloc[i+seq_len-1]
        close = merged['Close'].iloc[i+seq_len-1]
        next_close = merged['close_next'].iloc[i+seq_len-1]
        windows.append({"tech": tech, "text": text, "date": str(date), "close": float(close), "next_close": float(next_close)})
    return windows, feature_cols

def run_backtest(model_path, price_csv, news_csv=None, seq_len=30, device="cpu", initial_cash=10000, fee=0.0):
    windows, feature_cols = build_inference_windows(price_csv, news_csv, seq_len=seq_len)
    model = load_model(model_path, n_features=len(feature_cols), device=device)
    cash = initial_cash
    position = 0.0  # number of shares
    portfolio_values = []
    trades = []

    for w in windows:
        tech_tensor = torch.tensor(w['tech'][None, ...], dtype=torch.float32).to(device)
        enc = encode_headlines([w['text']], device=device)
        input_ids = enc['input_ids']
        attention_mask = enc['attention_mask']
        with torch.no_grad():
            logits = model(tech_tensor, input_ids, attention_mask)
            prob = torch.softmax(logits, dim=1)[0,1].cpu().item()  # prob of up
        # simple rule: go long if prob > 0.55, else flat
        if prob > 0.55 and position == 0:
            # buy as much as possible
            price = w['close']
            shares = (cash * (1 - fee)) / price
            position = shares
            cash -= shares * price * (1 + fee)
            trades.append({"date": w['date'], "action": "BUY", "price": price, "shares": shares, "prob": prob})
        elif prob <= 0.45 and position > 0:
            # sell all
            price = w['close']
            cash += position * price * (1 - fee)
            trades.append({"date": w['date'], "action": "SELL", "price": price, "shares": position, "prob": prob})
            position = 0

        total_value = cash + position * w['close']
        portfolio_values.append({"date": w['date'], "value": total_value})

    # finally liquidate
    if position > 0:
        last_price = windows[-1]['close']
        cash += position * last_price * (1 - fee)
        trades.append({"date": windows[-1]['date'], "action": "SELL_FINAL", "price": last_price, "shares": position, "prob": None})
        position = 0
    final_value = cash
    returns = (final_value - initial_cash) / initial_cash
    df_port = pd.DataFrame(portfolio_values)
    df_trades = pd.DataFrame(trades)
    return {"final_value": final_value, "returns": returns, "portfolio": df_port, "trades": df_trades}
