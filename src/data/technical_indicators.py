import pandas as pd
import ta

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # expects columns: date, Open, High, Low, Close, Adj Close (maybe), Volume
    df = df.copy().sort_values("date").reset_index(drop=True)
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    # Trend indicators
    df['ma5'] = close.rolling(5).mean()
    df['ma10'] = close.rolling(10).mean()
    df['ema12'] = close.ewm(span=12, adjust=False).mean()
    df['ema26'] = close.ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Volatility
    df['std5'] = close.rolling(5).std()
    df['atr14'] = ta.volatility.average_true_range(high, low, close, window=14)

    # Momentum
    df['rsi14'] = ta.momentum.rsi(close, window=14)
    df['stoch_k'] = ta.momentum.stoch(high, low, close, window=14, smooth_window=3)
    df['stoch_d'] = ta.momentum.stoch_signal(high, low, close, window=14, smooth_window=3)

    # Volume
    df['obv'] = ta.volume.on_balance_volume(close, volume)
    df['vwap'] = (df['Volume'] * (high + low + close) / 3).cumsum() / df['Volume'].cumsum()

    # Fill / drop
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.dropna(inplace=True)

    return df
