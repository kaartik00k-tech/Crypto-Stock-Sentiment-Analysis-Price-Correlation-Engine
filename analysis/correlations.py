from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def align_sentiment_returns(market: pd.DataFrame, sentiment: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """For lag=1, pair sentiment at session t with return at the next observed session."""
    if lag < 0:
        raise ValueError("lag must be >= 0")
    if market.empty:
        return pd.DataFrame(columns=["date", "mean_sentiment", "return_pct", "volume_change_pct"])
    m = market.copy()
    if "date" not in m:
        m["date"] = pd.to_datetime(m.index).strftime("%Y-%m-%d")
    m["date"] = pd.to_datetime(m["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    m = m.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    if "Close" not in m:
        raise ValueError("Market data must include a Close column")
    m["return_pct"] = pd.to_numeric(m["Close"], errors="coerce").pct_change() * 100
    if "Volume" in m:
        m["volume_change_pct"] = pd.to_numeric(m["Volume"], errors="coerce").pct_change() * 100
    else:
        m["volume_change_pct"] = np.nan
    s = sentiment.copy()
    if s.empty or not {"date", "mean_sentiment"}.issubset(s.columns):
        s = pd.DataFrame(columns=["date", "mean_sentiment"])
    else:
        s["date"] = pd.to_datetime(s["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        s = s.dropna(subset=["date"]).drop_duplicates("date")
    m["target_return"] = m["return_pct"].shift(-lag)
    m["target_volume_change"] = m["volume_change_pct"].shift(-lag)
    merged = m.merge(s[["date", "mean_sentiment"]], on="date", how="left")
    merged["return_pct"] = merged["target_return"]
    merged["volume_change_pct"] = merged["target_volume_change"]
    return merged[["date", "mean_sentiment", "return_pct", "volume_change_pct"]]


def correlation_stats(x, y, method: str = "Pearson", min_samples: int = 8) -> dict:
    """Return coefficient, p-value, and sample size after pairwise missing-value removal."""
    pairs = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(pairs)
    result = {"method": method.title(), "correlation": np.nan, "p_value": np.nan,
              "n": n, "significant_05": False}
    if n < min_samples or pairs["x"].nunique() < 2 or pairs["y"].nunique() < 2:
        return result
    fn = pearsonr if method.lower() == "pearson" else spearmanr if method.lower() == "spearman" else None
    if fn is None:
        raise ValueError("method must be Pearson or Spearman")
    coefficient, pvalue = fn(pairs["x"], pairs["y"])
    result.update(correlation=float(coefficient), p_value=float(pvalue),
                  significant_05=bool(pvalue < 0.05))
    return result


def lag_table(market: pd.DataFrame, sentiment: pd.DataFrame, lags=(0, 1, 2, 3, 5),
              method="Pearson", min_samples=8) -> pd.DataFrame:
    rows = []
    for lag in lags:
        aligned = align_sentiment_returns(market, sentiment, lag)
        for target in ("return_pct", "volume_change_pct"):
            stats = correlation_stats(aligned["mean_sentiment"], aligned[target], method, min_samples)
            rows.append({"lag_sessions": lag, "target": target, **stats})
    return pd.DataFrame(rows)
