from __future__ import annotations

from abc import ABC, abstractmethod

from quantitative_codex.execution.models import ExecutionReport, Order


class BrokerAdapter(ABC):
    @abstractmethod
    def submit_order(self, order: Order) -> ExecutionReport:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> ExecutionReport:
        raise NotImplementedError

    @abstractmethod
    def get_order(self, order_id: str) -> ExecutionReport:
        raise NotImplementedError
