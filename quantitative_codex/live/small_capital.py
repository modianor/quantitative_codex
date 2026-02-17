from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantitative_codex.execution.oms import OMS


@dataclass
class LiveTradingConfig:
    starting_equity: float = 10_000.0
    max_notional_per_order: float = 500.0
    max_daily_turnover_ratio: float = 0.25
    min_order_notional: float = 25.0


class SmallCapitalLiveRunner:
    """Paper-first small-capital execution guardrails.

    Converts target weights into capped target share quantities and routes them via OMS.
    """

    def __init__(self, oms: OMS, config: LiveTradingConfig | None = None) -> None:
        self.oms = oms
        self.config = config or LiveTradingConfig()

    def compute_target_shares(
        self,
        target_weights: pd.Series,
        prices: pd.Series,
        equity: float | None = None,
    ) -> pd.Series:
        eq = self.config.starting_equity if equity is None else equity
        tw = target_weights.reindex(prices.index).fillna(0.0)
        capped_notional = (tw * eq).clip(lower=-self.config.max_notional_per_order, upper=self.config.max_notional_per_order)

        valid_prices = prices.replace(0, pd.NA).astype(float)
        shares = (capped_notional / valid_prices).fillna(0.0)

        notional = (shares.abs() * prices).fillna(0.0)
        shares = shares.where(notional >= self.config.min_order_notional, 0.0)
        return shares

    def enforce_turnover_budget(self, current_shares: pd.Series, target_shares: pd.Series, prices: pd.Series, equity: float) -> pd.Series:
        aligned_current = current_shares.reindex(target_shares.index).fillna(0.0)
        aligned_prices = prices.reindex(target_shares.index).fillna(0.0)
        delta = target_shares - aligned_current

        gross_turnover = float((delta.abs() * aligned_prices).sum())
        budget = equity * self.config.max_daily_turnover_ratio
        if gross_turnover <= budget or gross_turnover <= 0:
            return target_shares

        scale = budget / gross_turnover
        return aligned_current + delta * scale

    def rebalance(self, target_weights: pd.Series, prices: pd.Series, equity: float | None = None) -> pd.Series:
        eq = self.config.starting_equity if equity is None else equity
        target_shares = self.compute_target_shares(target_weights, prices, equity=eq)
        current = self.oms.positions.snapshot().reindex(target_shares.index).fillna(0.0)
        budgeted = self.enforce_turnover_budget(current, target_shares, prices, eq)

        orders = self.oms.generate_orders_from_target(budgeted)
        self.oms.submit_orders(orders)
        return budgeted
