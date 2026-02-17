from __future__ import annotations

import pandas as pd


def build_postmortem_report(
    trades: pd.DataFrame,
    alerts: list[dict] | None = None,
    title: str = "Strategy Postmortem",
) -> dict[str, object]:
    """Create a lightweight post-run review payload for journaling."""
    required = {"symbol", "pnl"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"trades missing required columns: {sorted(missing)}")

    total_trades = int(len(trades))
    wins = int((trades["pnl"] > 0).sum())
    losses = int((trades["pnl"] < 0).sum())
    win_rate = float(wins / total_trades) if total_trades else 0.0
    gross_pnl = float(trades["pnl"].sum()) if total_trades else 0.0
    avg_pnl = float(trades["pnl"].mean()) if total_trades else 0.0

    by_symbol = (
        trades.groupby("symbol", as_index=False)["pnl"].sum().sort_values("pnl", ascending=False).to_dict("records")
        if total_trades
        else []
    )

    return {
        "title": title,
        "summary": {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "gross_pnl": gross_pnl,
            "avg_pnl": avg_pnl,
        },
        "top_symbols": by_symbol[:5],
        "alerts": alerts or [],
    }
