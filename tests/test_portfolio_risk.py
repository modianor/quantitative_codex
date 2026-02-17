import numpy as np
import pandas as pd

from quantitative_codex.portfolio import optimize_weights
from quantitative_codex.risk import apply_risk_controls, drawdown_guard


def test_optimize_weights_long_only_constraints():
    mu = pd.Series({"A": 0.1, "B": 0.05, "C": -0.01})
    risk = pd.Series({"A": 0.2, "B": 0.1, "C": 0.3})
    w = optimize_weights(mu, risk=risk, long_only=True, max_weight=0.7)

    assert set(w.index) == {"A", "B", "C"}
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()
    assert (w <= 0.7 + 1e-12).all()


def test_apply_risk_controls_and_drawdown_guard():
    w = pd.Series({"A": 0.8, "B": 0.2})
    adv = pd.Series({"A": 2e9, "B": 1e7})
    controlled = apply_risk_controls(w, adv_usd=adv, max_weight=0.6)
    assert np.isclose(controlled.sum(), 1.0)
    assert (controlled <= 1.0).all()

    equity_ok = pd.Series([1.0, 1.02, 1.01, 1.03])
    equity_bad = pd.Series([1.0, 0.9, 0.75])
    assert drawdown_guard(equity_ok, max_drawdown=0.3)
    assert not drawdown_guard(equity_bad, max_drawdown=0.2)
