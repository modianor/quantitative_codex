from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cleaning import add_adjusted_close, clean_eod_frame
from .schema import normalize_ohlcv_columns


def load_eod_csv(path: str | Path) -> pd.DataFrame:
    """Load vendor CSV and return normalized/cleaned EOD data."""
    df = pd.read_csv(path)
    df = normalize_ohlcv_columns(df)
    df = clean_eod_frame(df)
    df = add_adjusted_close(df)
    return df
