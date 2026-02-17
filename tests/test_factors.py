import numpy as np
import pandas as pd

from quantitative_codex.factors import compute_factor_library


def test_compute_factor_library_has_expected_columns():
    idx = pd.date_range("2023-01-01", periods=260, freq="B")
    close = pd.Series(np.linspace(100, 130, len(idx)), index=idx)
    volume = pd.Series(1_000_000 + np.arange(len(idx)), index=idx)
    df = pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "adj_close": close,
            "volume": volume,
        }
    )

    factors = compute_factor_library(df)
    expected = {
        "mom20",
        "mom60",
        "rev1",
        "realized_vol20",
        "price_sma50_ratio",
        "volume_z20",
        "rsi14",
        "macd_hist",
    }
    assert expected.issubset(factors.columns)
    assert factors.index.equals(df.index)
