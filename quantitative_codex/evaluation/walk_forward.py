from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantitative_codex.portfolio.optimization import optimize_weights
from quantitative_codex.risk.controls import apply_risk_controls


@dataclass
class WalkForwardConfig:
    train_window: int = 252 * 3
    test_window: int = 63
    rebalance_every: int = 5
    max_weight: float = 0.2
    one_way_bps: float = 2.0


def _annualized_metrics(returns: pd.Series) -> dict[str, float]:
    if returns.empty:
        return {"cagr": 0.0, "annual_vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    equity = (1 + returns).cumprod()
    n = len(returns)
    vol = float(returns.std(ddof=0) * np.sqrt(252))
    sharpe = float((returns.mean() / returns.std(ddof=0)) * np.sqrt(252)) if returns.std(ddof=0) > 0 else 0.0
    return {
        "cagr": float(equity.iloc[-1] ** (252 / n) - 1),
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
    }


def walk_forward_evaluate(
    prices: pd.DataFrame,
    config: WalkForwardConfig | None = None,
) -> dict[str, pd.DataFrame | pd.Series | dict[str, float]]:
    """Walk-forward evaluation for a cross-sectional momentum allocator.

    prices: wide DataFrame indexed by date, columns are symbols.
    """
    cfg = config or WalkForwardConfig()
    rets = prices.pct_change().fillna(0.0)

    all_returns: list[pd.Series] = []
    segment_rows: list[dict[str, float | int | str]] = []

    start = cfg.train_window
    while start + cfg.test_window <= len(prices):
        train_slice = slice(start - cfg.train_window, start)
        test_slice = slice(start, start + cfg.test_window)

        train_prices = prices.iloc[train_slice]
        train_rets = rets.iloc[train_slice]
        test_rets = rets.iloc[test_slice]

        # expected return proxy: trailing 60-day momentum on train end
        mom60 = train_prices.iloc[-1] / train_prices.iloc[-60] - 1
        risk20 = train_rets.tail(20).std(ddof=0).replace(0, np.nan).fillna(train_rets.std(ddof=0).median())

        base_w = optimize_weights(mom60, risk=risk20, long_only=True, max_weight=cfg.max_weight)

        adv_proxy = train_prices.iloc[-20:].mean() * 1_000_000  # simple placeholder ADV$ proxy
        w = apply_risk_controls(base_w, adv_usd=adv_proxy, max_weight=cfg.max_weight)

        # rebalance in test window on a fixed schedule
        test_period_returns = []
        current_w = w.reindex(test_rets.columns).fillna(0.0)
        for i, (_, row) in enumerate(test_rets.iterrows()):
            if i > 0 and i % cfg.rebalance_every == 0:
                rolling_train_prices = prices.iloc[start - cfg.train_window + i : start + i]
                rolling_train_rets = rets.iloc[start - cfg.train_window + i : start + i]
                if len(rolling_train_prices) >= 60:
                    mom60_roll = rolling_train_prices.iloc[-1] / rolling_train_prices.iloc[-60] - 1
                    risk20_roll = rolling_train_rets.tail(20).std(ddof=0).replace(0, np.nan).fillna(
                        rolling_train_rets.std(ddof=0).median()
                    )
                    base_w = optimize_weights(mom60_roll, risk=risk20_roll, long_only=True, max_weight=cfg.max_weight)
                    current_w = apply_risk_controls(base_w, max_weight=cfg.max_weight).reindex(test_rets.columns).fillna(0.0)

            turnover = float(current_w.abs().sum()) if i == 0 else 0.0
            gross_ret = float((current_w * row).sum())
            net_ret = gross_ret - turnover * (cfg.one_way_bps / 10000.0)
            test_period_returns.append(net_ret)

        seg_returns = pd.Series(test_period_returns, index=test_rets.index)
        seg_metrics = _annualized_metrics(seg_returns)

        segment_rows.append(
            {
                "start": str(test_rets.index[0].date()),
                "end": str(test_rets.index[-1].date()),
                "cagr": seg_metrics["cagr"],
                "annual_vol": seg_metrics["annual_vol"],
                "sharpe": seg_metrics["sharpe"],
                "max_drawdown": seg_metrics["max_drawdown"],
            }
        )
        all_returns.append(seg_returns)
        start += cfg.test_window

    if all_returns:
        returns = pd.concat(all_returns).sort_index()
    else:
        returns = pd.Series(dtype=float)

    equity = (1 + returns).cumprod() if not returns.empty else pd.Series(dtype=float)
    summary = _annualized_metrics(returns)

    return {
        "returns": returns,
        "equity": equity,
        "segments": pd.DataFrame(segment_rows),
        "summary": summary,
    }
