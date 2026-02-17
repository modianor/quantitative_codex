# Quantitative Codex MVP

This repository now includes a minimal daily-quant pipeline MVP with three layers:

1. **Data layer**: normalize, clean, and adjust EOD OHLCV data.
2. **Factor layer**: compute a compact factor library.
3. **Vectorized backtest layer**: run next-day execution simulation with explicit turnover costs.

## Quick start

```python
from quantitative_codex.data.providers import load_eod_csv
from quantitative_codex.factors.library import compute_factor_library
from quantitative_codex.backtest.vectorized import VectorizedBacktester

bars = load_eod_csv("sample.csv")
factors = compute_factor_library(bars)

# very simple signal: long when 20-day momentum is positive
signal = (factors["mom20"] > 0).astype(float)

bt = VectorizedBacktester(one_way_bps=2)
result = bt.run(bars["adj_close"], signal)
print(result.metrics)
```

## Notes

- Signals are shifted by 1 day before execution to avoid look-ahead bias.
- Cost model is a bps turnover model suitable for MVP research.
- This baseline can be extended to multi-asset cross-sectional workflows.
