import numpy as np
import pandas as pd

from quantitative_codex.backtest import VectorizedBacktester


def test_vectorized_backtester_runs_and_reports_metrics():
    idx = pd.date_range("2024-01-01", periods=120, freq="B")
    price = pd.Series(100 * (1 + 0.0005) ** np.arange(len(idx)), index=idx)
    signal = pd.Series(1.0, index=idx)

    result = VectorizedBacktester(one_way_bps=1.0).run(price, signal)

    assert set(result.metrics) == {"cagr", "annual_vol", "sharpe", "max_drawdown"}
    assert len(result.equity) == len(price)
    assert result.equity.iloc[-1] > 1.0
