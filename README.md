# Quantitative Codex MVP

This repository includes a minimal daily-quant pipeline with the following layers:

1. **Data layer**: normalize, clean, and adjust EOD OHLCV data.
2. **Factor layer**: compute a compact factor library.
3. **Vectorized backtest layer**: run next-day execution simulation with turnover costs.
4. **Portfolio + risk layer**: optimize weights and apply risk controls.
5. **Walk-forward evaluation**: rolling train/test performance evaluation.
6. **Execution + OMS + reconciliation**: paper broker adapter, order state tracking, and break checks.
7. **Live-ops toolkit**: small-capital live guardrails, monitoring alerts, parameter registry, and postmortem report generation.

## Quick start

```python
from quantitative_codex.data.providers import load_eod_csv
from quantitative_codex.factors.library import compute_factor_library
from quantitative_codex.backtest.vectorized import VectorizedBacktester

bars = load_eod_csv("sample.csv")
factors = compute_factor_library(bars)

signal = (factors["mom20"] > 0).astype(float)

bt = VectorizedBacktester(one_way_bps=2)
result = bt.run(bars["adj_close"], signal)
print(result.metrics)
```


## Single-stock backtest main function

```bash
python -m quantitative_codex.main   --csv ./data/AAPL.csv   --symbol AAPL   --strategy ma_cross   --cost-bps 2   --output ./artifacts/aapl_backtest.csv
```

Supported strategies: `mom20`, `ma_cross`, `rsi2_reversion`.

## Execution (paper) + OMS + reconciliation

```python
import pandas as pd
from quantitative_codex.execution.brokers import PaperBrokerAdapter
from quantitative_codex.execution.oms import OMS
from quantitative_codex.reconciliation import reconcile_positions

broker = PaperBrokerAdapter()
oms = OMS(broker)

orders = oms.generate_orders_from_target(pd.Series({"AAPL": 10.0}))
oms.submit_orders(orders)
broker.process_market_data("AAPL", 185.0)
oms.sync()

internal = oms.positions.snapshot()
broker_snapshot = pd.Series({"AAPL": 10.0})
breaks = reconcile_positions(internal, broker_snapshot)
print(breaks)
```

## Small-capital live run (paper-first)

```python
import pandas as pd
from quantitative_codex.live import LiveTradingConfig, SmallCapitalLiveRunner

cfg = LiveTradingConfig(starting_equity=5000, max_notional_per_order=300)
runner = SmallCapitalLiveRunner(oms, cfg)
runner.rebalance(
    target_weights=pd.Series({"AAPL": 0.10, "MSFT": 0.08}),
    prices=pd.Series({"AAPL": 190.0, "MSFT": 410.0}),
)
```

## Monitoring + parameter maintenance + review

```python
from quantitative_codex.monitoring import evaluate_alerts
from quantitative_codex.parameters import ParameterRegistry
from quantitative_codex.review import build_postmortem_report

registry = ParameterRegistry("./params.json")
registry.add("mom_trend", "v1.0.0", {"fast": 50, "slow": 200}, note="initial live")

alerts = evaluate_alerts(equity_curve, pnl_series, order_log_df, positions_notional)
report = build_postmortem_report(trades_df, alerts=[a.__dict__ for a in alerts])
```

## Notes

- Signals are shifted by 1 day before execution to avoid look-ahead bias.
- Cost model is a bps turnover model suitable for MVP research.
- Execution remains paper-first and deterministic by design for safe integration testing.
