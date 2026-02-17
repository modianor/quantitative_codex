from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    position: pd.Series
    turnover: pd.Series
    metrics: dict[str, float]


class VectorizedBacktester:
    def __init__(self, one_way_bps: float = 2.0):
        self.one_way_bps = one_way_bps

    def run(self, price: pd.Series, raw_signal: pd.Series) -> BacktestResult:
        ret = price.pct_change().fillna(0.0)
        pos = raw_signal.shift(1).fillna(0.0).clip(-1, 1)

        turnover = pos.diff().abs().fillna(pos.abs())
        cost = turnover * (self.one_way_bps / 10000.0)

        strat_ret = pos * ret - cost
        equity = (1 + strat_ret).cumprod()

        metrics = self._metrics(strat_ret, equity)
        return BacktestResult(
            equity=equity,
            returns=strat_ret,
            position=pos,
            turnover=turnover,
            metrics=metrics,
        )

    @staticmethod
    def _metrics(ret: pd.Series, equity: pd.Series) -> dict[str, float]:
        n = max(len(ret), 1)
        cagr = float(equity.iloc[-1] ** (252 / n) - 1)
        vol = float(ret.std(ddof=0) * np.sqrt(252))
        sharpe = float((ret.mean() / ret.std(ddof=0)) * np.sqrt(252)) if ret.std(ddof=0) > 0 else 0.0
        mdd = float((equity / equity.cummax() - 1).min())
        return {
            "cagr": cagr,
            "annual_vol": vol,
            "sharpe": sharpe,
            "max_drawdown": mdd,
        }
