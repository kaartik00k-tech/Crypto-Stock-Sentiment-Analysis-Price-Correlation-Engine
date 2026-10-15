# MarketPulse AI — Crypto & Stock Sentiment Analysis Engine

A Streamlit analytics app for exploring whether news sentiment is associated with asset returns or volume changes. It includes an explicitly simulated demo mode and a live-data mode using Yahoo Finance via `yfinance`, plus optional NewsAPI headlines.

> **Research tool, not financial advice.** Correlation does not prove causation or imply a profitable trading strategy. Market and news providers may be delayed, incomplete, unavailable, or subject to their own terms.

## Features

- BTC, ETH, SOL, AAPL, MSFT, NVDA, TSLA, and custom Yahoo Finance symbols.
- Demo mode that works without credentials or local data files; every simulated value is labeled.
- Live historical OHLCV market data via Yahoo Finance where supported.
- Optional live headlines via NewsAPI using `NEWSAPI_KEY`.
- VADER headline sentiment, daily sentiment aggregates, candlestick/volume chart, and lagged Pearson/Spearman correlations.
- Explicit prediction direction: lag 1 correlates sentiment on day *t* with the return on the next observed market session.
- Sample size and p-value displayed; minimum sample size is configurable.
- Unit tests for lag direction, correlation calculations, and sentiment aggregation.

## Run locally

Requires Python 3.11 or a compatible version supported by Streamlit Community Cloud.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The app opens in demo mode by default. No API keys are required to explore the dashboard.

## Live mode

1. Select **Live data** in the sidebar.
2. Choose a supported Yahoo Finance ticker and date range.
3. For news, create a key at [NewsAPI](https://newsapi.org/) and set `NEWSAPI_KEY` in your environment or hosting secrets.
4. Without a NewsAPI key, live market data remains usable; the app explains that live headlines are unavailable.

PowerShell example:

```powershell
$env:NEWSAPI_KEY="your_key_here"
streamlit run app.py
```

Yahoo Finance access is provided by a third-party community library and is not guaranteed. Custom symbols must be supported by Yahoo Finance. The application never silently replaces failed live data with synthetic data.

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/).
3. Choose **Create app**, select this repository and branch, and set the main file path to `app.py`.
4. Deploy. The app starts in demo mode without secrets.
5. For optional news, open the app's **Settings / Secrets** area and add:

   ```toml
   NEWSAPI_KEY = "your_key_here"
   ```

6. Restart/redeploy and switch to **Live data** in the sidebar.

Do not commit API keys, `.env`, or `.streamlit/secrets.toml`.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Methodology and limitations

- Demo mode uses a deterministic, generated dataset and synthetic headlines. These are for interface and pipeline demonstrations only.
- VADER is a general-purpose lexicon and may misread financial language, sarcasm, or context.
- NewsAPI availability and article coverage depend on plan and provider restrictions. The app only analyzes returned headlines; it does not scrape restricted pages.
- Market dates are aligned to the date labels returned by the market data provider. Crypto trades daily; equities have market-session gaps.
- Lag 0 compares sentiment and same-session return. Lag 1 compares sentiment with the next available market-session return; larger lags follow the same forward-session convention.
- P-values are exploratory and do not correct for multiple comparisons. Testing many assets/lags inflates false-positive risk.
- Correlations are descriptive and do not establish causal relationships or predictive performance.
