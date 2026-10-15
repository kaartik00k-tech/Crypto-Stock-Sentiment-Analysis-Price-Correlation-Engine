from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import requests


def fetch_headlines(query: str, start, end, api_key: str | None = None) -> pd.DataFrame:
    """Fetch headlines from NewsAPI. Returns an empty frame if no key is configured."""
    key = api_key or os.getenv("NEWSAPI_KEY")
    columns = ["published_at", "source", "text", "url"]
    if not key:
        return pd.DataFrame(columns=columns)
    params = {"q": query, "from": pd.Timestamp(start).strftime("%Y-%m-%d"),
              "to": pd.Timestamp(end).strftime("%Y-%m-%d"), "language": "en",
              "sortBy": "publishedAt", "pageSize": 100, "apiKey": key}
    try:
        response = requests.get("https://newsapi.org/v2/everything", params=params, timeout=15)
        if response.status_code == 429:
            raise ValueError("NewsAPI rate limit reached. Try again later.")
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise ValueError(f"News provider unavailable: {exc}") from exc
    if payload.get("status") != "ok":
        raise ValueError(payload.get("message", "News provider returned an unexpected response."))
    rows = []
    for article in payload.get("articles", []):
        if not article.get("title") or article.get("title") == "[Removed]":
            continue
        rows.append({"published_at": article.get("publishedAt"),
                     "source": (article.get("source") or {}).get("name", "Unknown"),
                     "text": article.get("title"), "url": article.get("url", "")})
    return pd.DataFrame(rows, columns=columns)
