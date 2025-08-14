#!/usr/bin/env bash
# Example run script. Update paths as needed.

TICKER=AAPL
START=2018-01-01
END=2024-12-31
DATA_DIR=./data

# 1) download prices (and optionally news -- requires NEWSAPI_KEY set in env)
python src/data/download_data.py --ticker $TICKER --start $START --end $END --fetch_news

# 2) train
python src/train.py --price_csv ${DATA_DIR}/price_${TICKER}.csv --news_csv ${DATA_DIR}/news_${TICKER}.csv --seq_len 30 --batch_size 8 --epochs 3

# 3) backtest
python - <<'PY'
from src.backtest.backtest import run_backtest
res = run_backtest("best_model.pt", price_csv="./data/price_AAPL.csv", news_csv="./data/news_AAPL.csv", seq_len=30, device="cpu")
print("Final value:", res["final_value"], "returns:", res["returns"])
print(res["trades"].head())
PY
