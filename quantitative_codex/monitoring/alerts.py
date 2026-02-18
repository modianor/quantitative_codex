from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Alert:
    level: str
    code: str
    message: str


@dataclass
class AlertRuleSet:
    max_drawdown: float = 0.1
    max_daily_loss: float = 0.03
    max_reject_ratio: float = 0.05
    max_position_abs: float = 1_000_000.0


def evaluate_alerts(
    equity_curve: pd.Series,
    pnl_series: pd.Series,
    order_log: pd.DataFrame,
    positions_notional: pd.Series,
    rules: AlertRuleSet | None = None,
) -> list[Alert]:
    cfg = rules or AlertRuleSet()
    alerts: list[Alert] = []

    if not equity_curve.empty:
        dd = float((equity_curve / equity_curve.cummax() - 1.0).min())
        if dd < -abs(cfg.max_drawdown):
            alerts.append(Alert("critical", "MAX_DRAWDOWN", f"drawdown breached: {dd:.2%}"))

    if not pnl_series.empty:
        daily_loss = float(pnl_series.min())
        if daily_loss < -abs(cfg.max_daily_loss):
            alerts.append(Alert("warning", "DAILY_LOSS", f"daily pnl breached: {daily_loss:.2%}"))

    if not order_log.empty and "status" in order_log.columns:
        rejected = (order_log["status"] == "rejected").sum()
        ratio = float(rejected / len(order_log))
        if ratio > cfg.max_reject_ratio:
            alerts.append(Alert("warning", "REJECT_RATIO", f"reject ratio too high: {ratio:.2%}"))

    if not positions_notional.empty:
        max_abs = float(positions_notional.abs().max())
        if max_abs > cfg.max_position_abs:
            alerts.append(Alert("critical", "POSITION_LIMIT", f"position abs notional breached: {max_abs:,.2f}"))

    return alerts
