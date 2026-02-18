from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    NEW = "new"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass
class Order:
    symbol: str
    qty: float
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    client_order_id: str | None = None


@dataclass
class Fill:
    symbol: str
    qty: float
    price: float
    side: OrderSide
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExecutionReport:
    order_id: str
    status: OrderStatus
    filled_qty: float
    avg_fill_price: float | None
    message: str = ""
