import pandas as pd

from quantitative_codex.reconciliation import reconcile_fills, reconcile_positions


def test_reconcile_positions_detects_breaks():
    internal = pd.Series({"AAPL": 10.0, "MSFT": 3.0})
    broker = pd.Series({"AAPL": 10.0, "MSFT": 2.0, "GOOG": 1.0})

    out = reconcile_positions(internal, broker, tolerance=1e-9)
    assert out.loc["AAPL", "is_break"] == False
    assert out.loc["MSFT", "is_break"] == True
    assert out.loc["GOOG", "is_break"] == True


def test_reconcile_fills_aggregates_and_flags_breaks():
    internal_fills = pd.DataFrame(
        [
            {"symbol": "AAPL", "side": "buy", "qty": 5.0},
            {"symbol": "AAPL", "side": "buy", "qty": 5.0},
            {"symbol": "MSFT", "side": "sell", "qty": 2.0},
        ]
    )
    broker_fills = pd.DataFrame(
        [
            {"symbol": "AAPL", "side": "buy", "qty": 10.0},
            {"symbol": "MSFT", "side": "sell", "qty": 1.0},
        ]
    )

    out = reconcile_fills(internal_fills, broker_fills, qty_tolerance=1e-9)
    msft = out[(out["symbol"] == "MSFT") & (out["side"] == "sell")].iloc[0]
    assert msft["is_break"]
