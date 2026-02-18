from __future__ import annotations

import numpy as np
import pandas as pd


def _cap_and_renormalize_long_only(weights: pd.Series, max_weight: float) -> pd.Series:
    w = weights.copy().clip(lower=0)
    if w.sum() == 0:
        return w
    w = w / w.sum()

    for _ in range(10):
        over = w > max_weight
        if not over.any():
            break
        capped_sum = w[over].clip(upper=max_weight).sum()
        residual = 1.0 - capped_sum
        free = ~over
        free_sum = w[free].sum()
        w.loc[over] = max_weight
        if free_sum <= 0 or residual <= 0:
            break
        w.loc[free] = w.loc[free] / free_sum * residual

    return w / w.sum() if w.sum() > 0 else w


def optimize_weights(
    expected_returns: pd.Series,
    risk: pd.Series | None = None,
    long_only: bool = True,
    max_weight: float = 0.2,
) -> pd.Series:
    """Simple constrained optimizer for MVP.

    Uses a risk-adjusted score (mu / risk) and projects into constraints.
    """
    mu = expected_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if mu.empty:
        return pd.Series(dtype=float)

    if risk is None:
        risk = pd.Series(1.0, index=mu.index)
    else:
        fallback = risk.median() if len(risk.dropna()) else 1.0
        risk = risk.reindex(mu.index).replace(0, np.nan).fillna(fallback)

    score = mu / risk
    if long_only:
        score = score.clip(lower=0)
        if score.sum() <= 0:
            w = pd.Series(1.0 / len(score), index=score.index)
        else:
            w = score / score.sum()
        return _cap_and_renormalize_long_only(w, max_weight).sort_index()

    if score.abs().sum() <= 0:
        return pd.Series(0.0, index=score.index)

    w = score / score.abs().sum()
    w = w.clip(lower=-max_weight, upper=max_weight)
    gross = w.abs().sum()
    return (w / gross if gross > 0 else w).sort_index()
