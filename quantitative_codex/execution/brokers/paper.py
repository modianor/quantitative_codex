from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from quantitative_codex.execution.brokers.base import BrokerAdapter
from quantitative_codex.execution.models import ExecutionReport, Fill, Order, OrderStatus, OrderType, OrderSide


@dataclass
class _OpenOrderState:
    order: Order
    remaining_qty: float
    filled_qty: float = 0.0
    weighted_notional: float = 0.0
    status: OrderStatus = OrderStatus.SUBMITTED


class PaperBrokerAdapter(BrokerAdapter):
    """Paper broker with deterministic fill logic.

    - Market orders fill immediately at the provided mark price.
    - Limit orders fill when mark price crosses the limit.
    """

    def __init__(self) -> None:
        self._orders: dict[str, _OpenOrderState] = {}
        self._fills: list[Fill] = []
        self._seq = 0

    def submit_order(self, order: Order) -> ExecutionReport:
        self._seq += 1
        order_id = order.client_order_id or f"paper-{self._seq:08d}"
        self._orders[order_id] = _OpenOrderState(order=order, remaining_qty=order.qty)
        return ExecutionReport(
            order_id=order_id,
            status=OrderStatus.SUBMITTED,
            filled_qty=0.0,
            avg_fill_price=None,
        )

    def cancel_order(self, order_id: str) -> ExecutionReport:
        state = self._orders.get(order_id)
        if state is None:
            return ExecutionReport(order_id=order_id, status=OrderStatus.REJECTED, filled_qty=0.0, avg_fill_price=None, message="order_not_found")

        if state.status in (OrderStatus.FILLED, OrderStatus.CANCELED):
            return self.get_order(order_id)

        state.status = OrderStatus.CANCELED
        return self.get_order(order_id)

    def get_order(self, order_id: str) -> ExecutionReport:
        state = self._orders.get(order_id)
        if state is None:
            return ExecutionReport(order_id=order_id, status=OrderStatus.REJECTED, filled_qty=0.0, avg_fill_price=None, message="order_not_found")

        avg = None if state.filled_qty <= 0 else state.weighted_notional / state.filled_qty
        return ExecutionReport(
            order_id=order_id,
            status=state.status,
            filled_qty=state.filled_qty,
            avg_fill_price=avg,
        )

    def process_market_data(self, symbol: str, mark_price: float, timestamp: datetime | None = None) -> list[ExecutionReport]:
        """Attempt fills against incoming mark price and return updated reports."""
        timestamp = timestamp or datetime.utcnow()
        updates: list[ExecutionReport] = []

        for order_id, state in self._orders.items():
            if state.status in (OrderStatus.CANCELED, OrderStatus.FILLED, OrderStatus.REJECTED):
                continue
            if state.order.symbol != symbol:
                continue

            fillable = False
            if state.order.order_type == OrderType.MARKET:
                fillable = True
            elif state.order.order_type == OrderType.LIMIT:
                if state.order.limit_price is None:
                    state.status = OrderStatus.REJECTED
                    updates.append(self.get_order(order_id))
                    continue
                if state.order.side == OrderSide.BUY and mark_price <= state.order.limit_price:
                    fillable = True
                if state.order.side == OrderSide.SELL and mark_price >= state.order.limit_price:
                    fillable = True

            if not fillable:
                continue

            fill_qty = state.remaining_qty
            state.filled_qty += fill_qty
            state.remaining_qty -= fill_qty
            state.weighted_notional += fill_qty * mark_price
            state.status = OrderStatus.FILLED if state.remaining_qty <= 0 else OrderStatus.PARTIALLY_FILLED

            self._fills.append(
                Fill(
                    symbol=state.order.symbol,
                    qty=fill_qty,
                    price=mark_price,
                    side=state.order.side,
                    timestamp=timestamp,
                )
            )
            updates.append(self.get_order(order_id))

        return updates

    @property
    def fills(self) -> list[Fill]:
        return list(self._fills)
