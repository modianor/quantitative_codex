import numpy as np
import pandas as pd

from quantitative_codex.evaluation import WalkForwardConfig, walk_forward_evaluate


def test_walk_forward_returns_expected_artifacts():
    idx = pd.date_range("2020-01-01", periods=900, freq="B")
    prices = pd.DataFrame(
        {
            "A": 100 * (1.0004 ** np.arange(len(idx))),
            "B": 90 * (1.0003 ** np.arange(len(idx))),
            "C": 110 * (1.0002 ** np.arange(len(idx))),
        },
        index=idx,
    )

    out = walk_forward_evaluate(
        prices,
        WalkForwardConfig(train_window=252, test_window=63, rebalance_every=5),
    )

    assert {"returns", "equity", "segments", "summary"}.issubset(out.keys())
    assert not out["segments"].empty
    assert "sharpe" in out["summary"]
