import unittest
import pandas as pd

from analysis.correlations import align_sentiment_returns, correlation_stats
from analysis.sentiment import aggregate_daily, score_text
from services.market_data import make_demo_market


class SentimentTests(unittest.TestCase):
    def test_score_has_label_and_compound(self):
        result = score_text("Company reports excellent growth")
        self.assertIn(result["label"], {"Positive", "Neutral", "Negative"})
        self.assertGreaterEqual(result["compound"], -1)
        self.assertLessEqual(result["compound"], 1)

    def test_daily_aggregation(self):
        items = pd.DataFrame([
            {"published_at": "2025-01-01T10:00:00Z", "text": "Excellent results and strong growth"},
            {"published_at": "2025-01-01T11:00:00Z", "text": "Severe losses and terrible outlook"},
        ])
        daily = aggregate_daily(items)
        self.assertEqual(len(daily), 1)
        self.assertEqual(int(daily.iloc[0]["item_count"]), 2)
        self.assertAlmostEqual(float(daily.iloc[0]["positive_pct"] + daily.iloc[0]["neutral_pct"] + daily.iloc[0]["negative_pct"]), 100.0)


class CorrelationTests(unittest.TestCase):
    def test_lag_direction_next_session(self):
        market = pd.DataFrame({"date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
                               "Close": [100, 110, 99, 118], "Volume": [10, 20, 30, 40]})
        sentiment = pd.DataFrame({"date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
                                  "mean_sentiment": [0.1, 0.2, 0.3, 0.4]})
        aligned = align_sentiment_returns(market, sentiment, lag=1)
        self.assertAlmostEqual(aligned.loc[0, "return_pct"], 10.0)
        self.assertAlmostEqual(aligned.loc[1, "return_pct"], -10.0)
        self.assertTrue(pd.isna(aligned.loc[3, "return_pct"]))

    def test_correlation_known_relationship(self):
        result = correlation_stats([1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12], min_samples=5)
        self.assertAlmostEqual(result["correlation"], 1.0)
        self.assertLess(result["p_value"], 0.05)
        self.assertEqual(result["n"], 6)

    def test_insufficient_samples(self):
        result = correlation_stats([1, 2], [1, 2], min_samples=5)
        self.assertTrue(pd.isna(result["correlation"]))
        self.assertEqual(result["n"], 2)

    def test_demo_data_is_deterministic(self):
        a = make_demo_market("AAPL", "2025-01-01", "2025-02-01")
        b = make_demo_market("AAPL", "2025-01-01", "2025-02-01")
        self.assertTrue(a["Close"].equals(b["Close"]))


if __name__ == "__main__":
    unittest.main()
