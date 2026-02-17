from __future__ import annotations

import pandas as pd


def reconcile_positions(internal_positions: pd.Series, broker_positions: pd.Series, tolerance: float = 1e-8) -> pd.DataFrame:
    """Compare internal OMS positions with broker positions."""
    idx = internal_positions.index.union(broker_positions.index)
    internal = internal_positions.reindex(idx).fillna(0.0)
    broker = broker_positions.reindex(idx).fillna(0.0)
    diff = internal - broker
    out = pd.DataFrame(
        {
            "internal_qty": internal,
            "broker_qty": broker,
            "diff_qty": diff,
            "is_break": diff.abs() > tolerance,
        }
    )
    return out.sort_index()


def reconcile_fills(internal_fills: pd.DataFrame, broker_fills: pd.DataFrame, qty_tolerance: float = 1e-8) -> pd.DataFrame:
    """Aggregate fill quantities by symbol/side and compare."""
    required_cols = {"symbol", "side", "qty"}
    missing_internal = required_cols - set(internal_fills.columns)
    missing_broker = required_cols - set(broker_fills.columns)
    if missing_internal:
        raise ValueError(f"internal_fills missing required columns: {sorted(missing_internal)}")
    if missing_broker:
        raise ValueError(f"broker_fills missing required columns: {sorted(missing_broker)}")

    int_agg = internal_fills.groupby(["symbol", "side"], as_index=False)["qty"].sum()
    brk_agg = broker_fills.groupby(["symbol", "side"], as_index=False)["qty"].sum()

    merged = int_agg.merge(brk_agg, on=["symbol", "side"], how="outer", suffixes=("_internal", "_broker")).fillna(0.0)
    merged["diff_qty"] = merged["qty_internal"] - merged["qty_broker"]
    merged["is_break"] = merged["diff_qty"].abs() > qty_tolerance
    return merged.sort_values(["symbol", "side"]).reset_index(drop=True)
