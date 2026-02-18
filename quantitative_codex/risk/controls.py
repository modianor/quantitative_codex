from __future__ import annotations

import pandas as pd


def liquidity_cap(weights: pd.Series, adv_usd: pd.Series, min_liquidity_score: float = 0.1) -> pd.Series:
    """Tilt down weights for less-liquid assets using normalized ADV score."""
    w = weights.copy()
    aligned_adv = adv_usd.reindex(w.index).fillna(0.0)
    max_adv = aligned_adv.max()
    if max_adv <= 0:
        return w

    liq_score = (aligned_adv / max_adv).clip(lower=min_liquidity_score, upper=1.0)
    w = w * liq_score
    if w.sum() > 0:
        w = w / w.sum()
    return w


def drawdown_guard(equity_curve: pd.Series, max_drawdown: float = 0.2) -> bool:
    """Return whether trading should remain enabled."""
    if equity_curve.empty:
        return True
    dd = (equity_curve / equity_curve.cummax() - 1.0).min()
    return dd >= -abs(max_drawdown)


def apply_risk_controls(
    weights: pd.Series,
    adv_usd: pd.Series | None = None,
    max_weight: float = 0.1,
    min_liquidity_score: float = 0.1,
) -> pd.Series:
    """Apply practical portfolio-level controls for MVP."""
    w = weights.copy().clip(lower=0, upper=max_weight)
    if w.sum() > 0:
        w = w / w.sum()

    if adv_usd is not None and not adv_usd.empty:
        w = liquidity_cap(w, adv_usd=adv_usd, min_liquidity_score=min_liquidity_score)

    return w
