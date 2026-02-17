from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

_COLUMN_ALIASES = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "date": "date",
    "Date": "date",
}


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common vendor column names to snake-case conventions."""
    out = df.rename(columns=_COLUMN_ALIASES).copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        out = out.set_index("date")
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()

    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return out
