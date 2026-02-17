from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_factor_library(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a compact MVP factor set from daily OHLCV."""
    price = df["adj_close"] if "adj_close" in df.columns else df["close"]
    ret_1d = price.pct_change()

    out = pd.DataFrame(index=df.index)
    out["mom20"] = price / price.shift(20) - 1
    out["mom60"] = price / price.shift(60) - 1
    out["rev1"] = -ret_1d
    out["realized_vol20"] = ret_1d.rolling(20).std() * np.sqrt(252)
    out["price_sma50_ratio"] = price / price.rolling(50).mean() - 1
    out["volume_z20"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()
    out["rsi14"] = rsi(price, n=14)

    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    out["macd_hist"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()

    return out


def cross_sectional_zscore(frame: pd.DataFrame, cap: float = 3.0) -> pd.DataFrame:
    """Z-score by row (date) for a panel where columns are symbols."""
    mu = frame.mean(axis=1)
    sigma = frame.std(axis=1).replace(0, np.nan)
    z = frame.sub(mu, axis=0).div(sigma, axis=0)
    return z.clip(lower=-cap, upper=cap)
