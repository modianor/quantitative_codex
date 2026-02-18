from __future__ import annotations

import numpy as np
import pandas as pd


def clean_eod_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Run minimal quality checks and remove clearly invalid rows."""
    out = df.copy()

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["open", "high", "low", "close"])

    valid_ohlc = (
        (out["low"] <= out["open"])
        & (out["low"] <= out["close"])
        & (out["high"] >= out["open"])
        & (out["high"] >= out["close"])
        & (out["high"] >= out["low"])
    )
    out = out.loc[valid_ohlc]

    if "volume" in out.columns:
        out = out[out["volume"] >= 0]

    return out


def add_adjusted_close(df: pd.DataFrame) -> pd.DataFrame:
    """Build adj_close from split/dividend fields when not provided.

    Formula (single-day approximation):
    adj_close = close / split_factor - div_cash
    """
    out = df.copy()
    if "adj_close" in out.columns:
        return out

    if "split_factor" in out.columns or "div_cash" in out.columns:
        split = out.get("split_factor", pd.Series(1.0, index=out.index)).replace(0, 1.0)
        div = out.get("div_cash", pd.Series(0.0, index=out.index)).fillna(0.0)
        out["adj_close"] = out["close"] / split - div
    else:
        out["adj_close"] = out["close"]

    return out
