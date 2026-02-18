from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from quantitative_codex.execution.brokers.base import BrokerAdapter
from quantitative_codex.execution.models import ExecutionReport, Order, OrderSide, OrderStatus, OrderType


@dataclass
class PositionBook:
    quantities: dict[str, float] = field(default_factory=dict)

    def apply_fill(self, symbol: str, qty: float, side: OrderSide) -> None:
        signed = qty if side == OrderSide.BUY else -qty
        self.quantities[symbol] = self.quantities.get(symbol, 0.0) + signed

    def snapshot(self) -> pd.Series:
        if not self.quantities:
            return pd.Series(dtype=float)
        return pd.Series(self.quantities).sort_index()


class OMS:
    """Simple order management system for target-position execution."""

    def __init__(self, broker: BrokerAdapter) -> None:
        self.broker = broker
        self.positions = PositionBook()
        self.order_log: list[dict[str, object]] = []
        self._open_orders: dict[str, Order] = {}
        self._applied_fills: dict[str, float] = {}

    def generate_orders_from_target(self, target_positions: pd.Series) -> list[Order]:
        current = self.positions.snapshot().reindex(target_positions.index).fillna(0.0)
        delta = target_positions - current
        orders: list[Order] = []

        for symbol, d in delta.items():
            if abs(d) < 1e-12:
                continue
            side = OrderSide.BUY if d > 0 else OrderSide.SELL
            orders.append(
                Order(
                    symbol=symbol,
                    qty=abs(float(d)),
                    side=side,
                    order_type=OrderType.MARKET,
                )
            )
        return orders

    def submit_orders(self, orders: list[Order]) -> list[ExecutionReport]:
        reports = []
        for order in orders:
            report = self.broker.submit_order(order)
            reports.append(report)
            if report.status in (OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED):
                self._open_orders[report.order_id] = order
                self._applied_fills.setdefault(report.order_id, 0.0)
            self._log_event("submit", report.order_id, order.symbol, order.qty, order.side.value, report.status.value)
        return reports

    def sync(self) -> list[ExecutionReport]:
        updates = []
        for order_id, order in list(self._open_orders.items()):
            report = self.broker.get_order(order_id)
            updates.append(report)
            self._log_event("status", order_id, order.symbol, order.qty, order.side.value, report.status.value)

            prev_applied = self._applied_fills.get(order_id, 0.0)
            incremental_fill = max(report.filled_qty - prev_applied, 0.0)
            if incremental_fill > 0:
                self.positions.apply_fill(order.symbol, incremental_fill, order.side)
                self._applied_fills[order_id] = report.filled_qty

            if report.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
                self._open_orders.pop(order_id, None)

        return updates

    def _log_event(self, event: str, order_id: str, symbol: str, qty: float, side: str, status: str) -> None:
        self.order_log.append(
            {
                "ts": datetime.utcnow().isoformat(),
                "event": event,
                "order_id": order_id,
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "status": status,
            }
        )
