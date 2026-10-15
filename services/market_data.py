from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_market_data(ticker: str, start, end) -> pd.DataFrame:
    """Fetch real OHLCV data. Raises a readable ValueError on provider failures."""
    symbol = str(ticker or "").strip().upper()
    if not symbol or len(symbol) > 30 or any(ch.isspace() for ch in symbol):
        raise ValueError("Enter a valid ticker symbol (for example AAPL or BTC-USD).")
    try:
        frame = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False,
                            threads=False, timeout=15)
    except Exception as exc:
        raise ValueError(f"Market data provider request failed: {exc}") from exc
    if frame is None or frame.empty:
        raise ValueError(f"No market data returned for {symbol}. Check the ticker, date range, or provider availability.")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    needed = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in frame.columns]
    if "Close" not in needed:
        raise ValueError(f"Provider returned no closing prices for {symbol}.")
    frame = frame[needed].copy()
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(None)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame["return_pct"] = frame["Close"].pct_change() * 100
    frame["volatility_20d"] = frame["return_pct"].rolling(20, min_periods=5).std()
    frame["date"] = frame.index.strftime("%Y-%m-%d")
    return frame


def make_demo_market(ticker: str, start, end) -> pd.DataFrame:
    """Deterministic synthetic prices, permanently identified as demo data by the UI."""
    dates = pd.date_range(start=start, end=end, freq="D")
    if len(dates) < 3:
        dates = pd.date_range(end=pd.Timestamp(end), periods=30, freq="D")
    seed = sum(ord(c) for c in str(ticker)) + len(dates)
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.018, len(dates))
    close = 100 * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, 0.004, len(dates)))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.015, len(dates)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.015, len(dates)))
    volume = rng.integers(100_000, 2_000_000, len(dates))
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                       "Volume": volume}, index=dates)
    df["return_pct"] = df["Close"].pct_change() * 100
    df["volatility_20d"] = df["return_pct"].rolling(20, min_periods=5).std()
    df["date"] = df.index.strftime("%Y-%m-%d")
    return df
