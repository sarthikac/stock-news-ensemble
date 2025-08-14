# src/data/download_data.py
import os
import argparse
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
import requests
from time import sleep

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./data")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

os.makedirs(DATA_DIR, exist_ok=True)

def download_prices(ticker, start, end, interval="1d"):
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    df = df.reset_index()
    df.rename(columns={"Date": "date"}, inplace=True)
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    return df

def fetch_news_newsapi(query, from_dt, to_dt, page=1):
    """Fetch headlines using NewsAPI (free plan limits)."""
    if not NEWSAPI_KEY:
        raise RuntimeError("NEWSAPI_KEY not set in environment; set it or provide your own headlines CSV.")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_dt,
        "to": to_dt,
        "pageSize": 100,
        "page": page,
        "language": "en",
        "sortBy": "relevancy",
        "apiKey": NEWSAPI_KEY
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def download_news_for_period(query, start, end):
    # NewsAPI free plan restricts history; for a real system use paid or other sources.
    all_articles = []
    # iterate in 7-day windows to be gentler
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    period = timedelta(days=7)
    current = start_dt
    while current <= end_dt:
        window_end = min(current + period, end_dt)
        from_dt = current.strftime("%Y-%m-%d")
        to_dt = window_end.strftime("%Y-%m-%d")
        try:
            res = fetch_news_newsapi(query, from_dt, to_dt)
        except Exception as e:
            print("NewsAPI fetch error:", e)
            break
        articles = res.get("articles", [])
        for a in articles:
            all_articles.append({"date": a["publishedAt"][:10], "headline": a["title"]})
        sleep(1.0)  # be polite
        current += period
    df = pd.DataFrame(all_articles)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--fetch_news", action="store_true")
    args = parser.parse_args()

    prices = download_prices(args.ticker, args.start, args.end)
    prices.to_csv(os.path.join(DATA_DIR, f"price_{args.ticker}.csv"), index=False)
    print("Saved prices:", prices.shape)

    if args.fetch_news:
        print("Fetching news (may be limited by NewsAPI history)...")
        news = download_news_for_period(args.ticker, args.start, args.end)
        news.to_csv(os.path.join(DATA_DIR, f"news_{args.ticker}.csv"), index=False)
        print("Saved news:", news.shape)
    else:
        print("Skip fetching news. Provide news CSV manually if needed.")

if __name__ == "__main__":
    main()

