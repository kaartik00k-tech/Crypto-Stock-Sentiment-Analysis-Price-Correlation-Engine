from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analysis.correlations import align_sentiment_returns, correlation_stats, lag_table
from analysis.sentiment import aggregate_daily, score_text
from services.market_data import fetch_market_data, make_demo_market
from services.news_data import fetch_headlines

st.set_page_config(page_title="MarketPulse AI", page_icon="📈", layout="wide")

st.markdown("""
<style>
.stApp {background: #0b1020; color: #e8edf7;}
[data-testid="stHeader"] {background: rgba(11,16,32,.9);}
[data-testid="stMetric"] {background:#131b2e; border:1px solid #26334d; padding:16px; border-radius:12px;}
div[data-testid="stAlert"] {border-radius:10px;}
h1,h2,h3 {letter-spacing:-.025em;}
.small-muted {color:#9caac2;font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

st.title("📈 MarketPulse AI")
st.caption("Crypto & Stock Sentiment Analysis Engine · Research dashboard")

with st.sidebar:
    st.header("Analysis settings")
    mode = st.radio("Data mode", ["Demo mode", "Live data"], help="Demo data is synthetic and is never substituted for failed live requests.")
    category = st.selectbox("Asset category", ["Cryptocurrency", "Stock", "Custom ticker"])
    symbols = {
        "Cryptocurrency": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD"},
        "Stock": {"Apple": "AAPL", "Microsoft": "MSFT", "NVIDIA": "NVDA", "Tesla": "TSLA"},
    }
    if category == "Custom ticker":
        ticker = st.text_input("Yahoo Finance ticker", "AAPL").strip().upper()
    else:
        selected = st.selectbox("Asset", list(symbols[category]))
        ticker = symbols[category][selected]
    default_start = date.today() - timedelta(days=180)
    start_date = st.date_input("Start date", value=default_start, max_value=date.today() - timedelta(days=2))
    end_date = st.date_input("End date", value=date.today() - timedelta(days=1), max_value=date.today())
    method = st.selectbox("Correlation method", ["Pearson", "Spearman"])
    lag = st.selectbox("Focus lag (market sessions)", [0, 1, 2, 3, 5], index=1)
    min_samples = st.slider("Minimum paired observations", min_value=5, max_value=30, value=8)
    run = st.button("Run analysis", type="primary", use_container_width=True)

if start_date >= end_date:
    st.error("Choose a start date earlier than the end date.")
    st.stop()

if "analysis_key" not in st.session_state:
    st.session_state.analysis_key = None
if run or st.session_state.analysis_key is None:
    st.session_state.analysis_key = f"{mode}|{ticker}|{start_date}|{end_date}|{method}|{lag}|{min_samples}"

@st.cache_data(ttl=900, show_spinner=False)
def get_live_market(symbol: str, start: date, end: date):
    return fetch_market_data(symbol, start, end + timedelta(days=1))

@st.cache_data(ttl=900, show_spinner=False)
def get_live_news(symbol: str, start: date, end: date, api_key: str):
    return fetch_headlines(symbol, start, end, api_key or None)

if mode == "Demo mode":
    market = make_demo_market(ticker, start_date, end_date)
    rng = np.random.default_rng(sum(ord(c) for c in ticker) + len(market))
    phrases = [
        ("Investors welcome stronger growth outlook", 0.45),
        ("Market faces uncertainty amid mixed signals", -0.25),
        ("Analysts remain cautious on near-term risks", -0.35),
        ("Company reports encouraging product momentum", 0.5),
        ("Trading remains steady as investors assess news", 0.05),
        ("Sector outlook improves after positive update", 0.4),
        ("Market volatility rises amid investor concerns", -0.4),
    ]
    rows = []
    for d in market.index:
        if rng.random() < 0.72:
            headline, _ = phrases[int(rng.integers(0, len(phrases)))]
            rows.append({"published_at": d.strftime("%Y-%m-%dT15:00:00Z"),
                         "source": "SIMULATED SOURCE", "text": f"[SIMULATED] {ticker}: {headline}",
                         "url": ""})
    news = pd.DataFrame(rows, columns=["published_at", "source", "text", "url"])
    st.warning("DEMO MODE — all prices, headlines, and sentiment observations are synthetic. They are not real market data.")
else:
    try:
        with st.spinner("Fetching market data..."):
            market = get_live_market(ticker, start_date, end_date)
    except Exception as exc:
        st.error(f"Live market data could not be loaded. No synthetic fallback was used. Details: {exc}")
        st.stop()
    try:
        key = st.secrets.get("NEWSAPI_KEY", os.getenv("NEWSAPI_KEY", ""))
    except Exception:
        key = os.getenv("NEWSAPI_KEY", "")
    try:
        with st.spinner("Fetching available headlines..."):
            news = get_live_news(ticker, start_date, end_date, key)
    except Exception as exc:
        st.warning(f"Live market data is available, but headlines could not be loaded: {exc}")
        news = pd.DataFrame(columns=["published_at", "source", "text", "url"])
    st.info("Live market data is provided by Yahoo Finance via yfinance. News is provided by NewsAPI only when NEWSAPI_KEY is configured; coverage and freshness vary by provider.")

if market.empty:
    st.error("No market observations are available for this date range.")
    st.stop()

daily = aggregate_daily(news)
if not daily.empty:
    news = news.copy()
    scored = news["text"].fillna("").map(score_text).apply(pd.Series)
    news["sentiment"] = scored["label"]
    news["compound"] = scored["compound"]
    news["published_at"] = pd.to_datetime(news["published_at"], utc=True, errors="coerce")
else:
    news = pd.DataFrame(columns=["published_at", "source", "text", "url", "sentiment", "compound"])

last_close = float(pd.to_numeric(market["Close"], errors="coerce").dropna().iloc[-1])
returns = pd.to_numeric(market["Close"], errors="coerce").pct_change().dropna()
last_return = float(returns.iloc[-1] * 100) if len(returns) else float("nan")
last_volume = float(market["Volume"].iloc[-1]) if "Volume" in market and pd.notna(market["Volume"].iloc[-1]) else float("nan")
latest_sent = float(daily["mean_sentiment"].iloc[-1]) if not daily.empty else float("nan")
n_news = int(len(news))

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Latest close", f"{last_close:,.2f}")
m2.metric("Last session return", f"{last_return:+.2f}%" if np.isfinite(last_return) else "N/A")
m3.metric("Latest volume", f"{last_volume:,.0f}" if np.isfinite(last_volume) else "N/A")
m4.metric("Latest daily sentiment", f"{latest_sent:+.3f}" if np.isfinite(latest_sent) else "No news")
m5.metric("Headlines analyzed", f"{n_news:,}")

st.caption(f"Selected asset: {ticker} · Market observations: {len(market)} · Data mode: {mode} · Last market date: {market.index[-1].strftime('%Y-%m-%d')}")

st.subheader("Market price & volume")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                    row_heights=[0.72, 0.28], specs=[[{"secondary_y": False}], [{"secondary_y": False}]])
fig.add_trace(go.Candlestick(x=market.index, open=market["Open"] if "Open" in market else market["Close"],
    high=market["High"] if "High" in market else market["Close"],
    low=market["Low"] if "Low" in market else market["Close"], close=market["Close"], name="OHLC"), row=1, col=1)
if "Volume" in market:
    fig.add_trace(go.Bar(x=market.index, y=market["Volume"], name="Volume", marker_color="#526a91"), row=2, col=1)
fig.update_layout(template="plotly_dark", height=540, margin=dict(l=10,r=10,t=25,b=10),
                  xaxis_rangeslider_visible=False, legend_orientation="h")
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Daily sentiment timeline")
    if daily.empty:
        st.info("No headline sentiment is available for this period. In live mode, configure NEWSAPI_KEY to request headlines. Missing news is not treated as neutral sentiment.")
    else:
        sf = go.Figure()
        sf.add_trace(go.Bar(x=daily["date"], y=daily["positive_pct"], name="Positive %"))
        sf.add_trace(go.Bar(x=daily["date"], y=daily["neutral_pct"], name="Neutral %"))
        sf.add_trace(go.Bar(x=daily["date"], y=daily["negative_pct"], name="Negative %"))
        sf.update_layout(barmode="stack", template="plotly_dark", height=350, yaxis_title="Share of headlines (%)",
                         margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(sf, use_container_width=True)
with right:
    st.subheader(f"Sentiment vs return · lag {lag}")
    aligned = align_sentiment_returns(market, daily, lag)
    paired = aligned.dropna(subset=["mean_sentiment", "return_pct"])
    if paired.empty:
        st.info("No overlapping sentiment and return observations are available for this lag.")
    else:
        sc = go.Figure(go.Scatter(x=paired["mean_sentiment"], y=paired["return_pct"], mode="markers",
                                  marker=dict(size=8, opacity=.75), text=paired["date"],
                                  hovertemplate="Date: %{text}<br>Sentiment: %{x:.3f}<br>Target return: %{y:.2f}%<extra></extra>"))
        sc.update_layout(template="plotly_dark", height=350, xaxis_title="Sentiment on source day",
                         yaxis_title="Return on target session (%)", margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(sc, use_container_width=True)

st.subheader("Lagged correlation analysis")
st.caption("Direction: lag 1 pairs sentiment on date t with the return on the next observed market session. Lag 0 uses the same session; larger lags use later sessions. P-values are exploratory and do not adjust for multiple testing.")
if daily.empty:
    st.info("Correlation analysis requires actual or demo headline sentiment. Missing sentiment days are excluded, not filled with zero.")
else:
    table = lag_table(market, daily, method=method, min_samples=min_samples)
    display = table.copy()
    display["correlation"] = display["correlation"].map(lambda x: round(x, 4) if pd.notna(x) else None)
    display["p_value"] = display["p_value"].map(lambda x: round(x, 4) if pd.notna(x) else None)
    display["significant_05"] = display["significant_05"].map({True: "Yes (uncorrected)", False: "No / insufficient data"})
    st.dataframe(display, use_container_width=True, hide_index=True)
    chosen = table[(table["lag_sessions"] == lag) & (table["target"] == "return_pct")].iloc[0]
    if pd.isna(chosen["correlation"]):
        st.warning(f"Insufficient valid pairs for the selected lag. At least {min_samples} paired observations and variation in both series are required.")
    else:
        st.write(f"At lag {lag}, sentiment and target returns have a {chosen['correlation']:+.3f} {method.lower()} correlation based on {int(chosen['n'])} paired observations (p = {chosen['p_value']:.4f}).")
        if chosen["significant_05"]:
            st.caption("The unadjusted p-value is below 0.05; this is exploratory evidence only, especially when multiple lags/assets are tested.")
        else:
            st.caption("The association is not statistically significant at the 0.05 threshold, or the sample is insufficient.")
    st.info("Interpretation: correlation is not causation. These descriptive results do not prove sentiment predicts prices or establish a profitable strategy.")

st.subheader("Collected headlines")
if news.empty:
    st.info("No headlines are available. Demo mode generates clearly labeled synthetic headlines; live mode requires a configured news provider and may return no articles.")
else:
    view = news.copy()
    view["published_at"] = pd.to_datetime(view["published_at"], utc=True, errors="coerce").astype(str)
    view["sentiment_score"] = view.get("compound", pd.Series(dtype=float))
    view["sentiment_label"] = view.get("sentiment", pd.Series(dtype=str))
    for _, row in view.sort_values("published_at", ascending=False).head(30).iterrows():
        title = str(row.get("text", "Untitled"))
        source = str(row.get("source", "Unknown"))
        stamp = str(row.get("published_at", "Unknown time"))
        label = str(row.get("sentiment_label", "Unscored"))
        score = row.get("sentiment_score", np.nan)
        link = str(row.get("url", ""))
        if link and link.startswith("https://"):
            st.markdown(f"- [{title}]({link})  \n  <span class='small-muted'>{source} · {stamp} · {label} · {score:+.3f}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"- **{title}**  \n  <span class='small-muted'>{source} · {stamp} · {label} · {score:+.3f}</span>", unsafe_allow_html=True)

st.divider()
st.caption("MarketPulse AI · Educational research software. Verify provider terms and data timestamps before relying on results.")
