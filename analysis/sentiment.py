from __future__ import annotations

import re
from typing import Iterable

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_ANALYZER = SentimentIntensityAnalyzer()


def score_text(text: str) -> dict[str, float | str]:
    """Score a headline with VADER; empty text is treated as neutral."""
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    scores = _ANALYZER.polarity_scores(clean) if clean else {
        "neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0
    }
    label = "Positive" if scores["compound"] >= 0.05 else (
        "Negative" if scores["compound"] <= -0.05 else "Neutral"
    )
    return {**scores, "label": label, "text": clean}


def aggregate_daily(items: pd.DataFrame) -> pd.DataFrame:
    """Aggregate text items by UTC calendar day. Empty days remain absent, not neutral."""
    cols = ["date", "mean_sentiment", "positive_pct", "neutral_pct",
            "negative_pct", "item_count", "sentiment_change"]
    if items.empty or not {"published_at", "text"}.issubset(items.columns):
        return pd.DataFrame(columns=cols)
    df = items.copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].ne("")]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"]).drop_duplicates(subset=["text", "published_at"])
    if df.empty:
        return pd.DataFrame(columns=cols)
    scored = df["text"].map(score_text).apply(pd.Series)
    df["compound"] = scored["compound"].astype(float)
    df["label"] = scored["label"]
    df["date"] = df["published_at"].dt.strftime("%Y-%m-%d")
    daily = df.groupby("date", as_index=False).agg(
        mean_sentiment=("compound", "mean"),
        item_count=("compound", "size"),
        positive_pct=("label", lambda s: (s == "Positive").mean() * 100),
        neutral_pct=("label", lambda s: (s == "Neutral").mean() * 100),
        negative_pct=("label", lambda s: (s == "Negative").mean() * 100),
    )
    daily["sentiment_change"] = daily["mean_sentiment"].diff()
    return daily[cols]
