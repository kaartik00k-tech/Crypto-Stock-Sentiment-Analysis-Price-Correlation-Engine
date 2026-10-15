"""Social data adapter boundary.

This starter project intentionally does not scrape social platforms. Add a provider only
when you have approved API credentials and comply with that provider's terms, quotas,
and data-use policies.
"""

import pandas as pd


def empty_social_items() -> pd.DataFrame:
    return pd.DataFrame(columns=["published_at", "source", "text", "url"])
