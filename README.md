# Quantitative Codex MVP

This repository includes a minimal daily-quant pipeline with the following layers:

1. **Data layer**: normalize, clean, and adjust EOD OHLCV data.
2. **Factor layer**: compute a compact factor library.
3. **Vectorized backtest layer**: run next-day execution simulation with turnover costs.
4. **Portfolio + risk layer**: optimize weights and apply risk controls.
5. **Walk-forward evaluation**: rolling train/test performance evaluation.

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

## Portfolio optimization + risk controls

```python
import pandas as pd
from quantitative_codex.portfolio import optimize_weights
from quantitative_codex.risk import apply_risk_controls

mu = pd.Series({"AAPL": 0.10, "MSFT": 0.08, "AMZN": 0.06})
risk = pd.Series({"AAPL": 0.20, "MSFT": 0.18, "AMZN": 0.24})
adv = pd.Series({"AAPL": 5e9, "MSFT": 4e9, "AMZN": 3e9})

weights = optimize_weights(mu, risk=risk, max_weight=0.5)
weights = apply_risk_controls(weights, adv_usd=adv, max_weight=0.5)
```

## Walk-forward evaluation

```python
from quantitative_codex.evaluation import WalkForwardConfig, walk_forward_evaluate

# prices: wide DataFrame [date x symbols]
wf = walk_forward_evaluate(
    prices,
    WalkForwardConfig(train_window=756, test_window=63, rebalance_every=5),
)
print(wf["summary"])
print(wf["segments"].tail())
```

## Notes

- Signals are shifted by 1 day before execution to avoid look-ahead bias.
- Cost model is a bps turnover model suitable for MVP research.
- The walk-forward module is intentionally simple and intended as a baseline for extension.
