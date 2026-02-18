from __future__ import annotations

import argparse
import io
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quantitative_codex.backtest.vectorized import BacktestResult, VectorizedBacktester
from quantitative_codex.data.providers import load_eod_csv
from quantitative_codex.factors.library import compute_factor_library, rsi


@dataclass
class SingleStockBacktestOutput:
    symbol: str
    strategy: str
    result: BacktestResult
    frame: pd.DataFrame


def fetch_stooq_daily(symbol: str) -> pd.DataFrame:
    """Download daily OHLCV bars from Stooq for a US symbol (e.g., NVDA)."""
    ticker = symbol.lower().replace(".us", "") + ".us"
    query = urllib.parse.urlencode({"s": ticker, "i": "d"})
    url = f"https://stooq.com/q/d/l/?{query}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = resp.read().decode("utf-8")

    df = pd.read_csv(io.StringIO(payload))
    if df.empty:
        raise ValueError(f"No data returned from Stooq for symbol={symbol}")
    return df


def get_bars(csv_path: str | Path | None, symbol: str | None) -> pd.DataFrame:
    if csv_path:
        return load_eod_csv(csv_path)
    if not symbol:
        raise ValueError("Either --csv or --symbol must be provided")

    raw = fetch_stooq_daily(symbol)
    cache_dir = Path(".cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{symbol.upper()}_stooq_daily.csv"
    raw.to_csv(cache_file, index=False)
    return load_eod_csv(cache_file)


def build_signal(df: pd.DataFrame, strategy: str = "mom20") -> pd.Series:
    """Generate a single-stock daily signal in [-1, 1]."""
    strategy = strategy.lower()
    price = df["adj_close"] if "adj_close" in df.columns else df["close"]

    if strategy == "mom20":
        factors = compute_factor_library(df)
        return (factors["mom20"] > 0).astype(float)

    if strategy == "ma_cross":
        fast = price.rolling(50).mean()
        slow = price.rolling(200).mean()
        return (fast > slow).astype(float)

    if strategy == "rsi2_reversion":
        rsi2 = rsi(price, n=2)
        sig = pd.Series(0.0, index=price.index)
        sig = sig.mask(rsi2 < 10, 1.0)
        sig = sig.mask(rsi2 > 60, 0.0)
        return sig.ffill().fillna(0.0)

    raise ValueError(f"Unsupported strategy: {strategy}")


def run_single_stock_backtest(
    csv_path: str | Path | None = None,
    symbol: str = "UNKNOWN",
    strategy: str = "mom20",
    one_way_bps: float = 2.0,
) -> SingleStockBacktestOutput:
    """Run a daily single-stock vectorized backtest from local CSV or auto-downloaded data."""
    bars = get_bars(csv_path=csv_path, symbol=symbol)
    signal = build_signal(bars, strategy=strategy)
    price = bars["adj_close"] if "adj_close" in bars.columns else bars["close"]

    result = VectorizedBacktester(one_way_bps=one_way_bps).run(price=price, raw_signal=signal)

    frame = pd.DataFrame(
        {
            "close": bars["close"],
            "adj_close": price,
            "signal": signal,
            "position": result.position,
            "strategy_ret": result.returns,
            "equity": result.equity,
        }
    )

    return SingleStockBacktestOutput(symbol=symbol, strategy=strategy, result=result, frame=frame)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-stock EOD backtest")
    parser.add_argument("--csv", default="", help="Path to OHLCV csv (optional if --symbol is provided)")
    parser.add_argument("--symbol", default="", help="Ticker symbol label (e.g., NVDA). Used for auto-download if --csv absent")
    parser.add_argument(
        "--strategy",
        default="mom20",
        choices=["mom20", "ma_cross", "rsi2_reversion"],
        help="Single-stock signal generator",
    )
    parser.add_argument("--cost-bps", type=float, default=2.0, help="One-way cost in bps")
    parser.add_argument("--output", default="", help="Optional output csv path for detailed series")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = run_single_stock_backtest(
        csv_path=args.csv or None,
        symbol=(args.symbol or "UNKNOWN"),
        strategy=args.strategy,
        one_way_bps=args.cost_bps,
    )

    print(f"symbol={out.symbol} strategy={out.strategy}")
    print({k: round(v, 6) for k, v in out.result.metrics.items()})

    if args.output:
        out.frame.to_csv(args.output, index=True)
        print(f"saved_detail={args.output}")


if __name__ == "__main__":
    main()
