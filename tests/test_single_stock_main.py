import pandas as pd

from quantitative_codex.main import build_signal, run_single_stock_backtest


def test_build_signal_mom20_shape():
    idx = pd.date_range("2024-01-01", periods=260, freq="B")
    close = pd.Series(range(100, 360), index=idx, dtype=float)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "adj_close": close,
            "volume": 1_000_000.0,
        }
    )
    sig = build_signal(df, strategy="mom20")
    assert sig.index.equals(df.index)


def test_run_single_stock_backtest_from_csv(tmp_path):
    csv = tmp_path / "bars.csv"
    csv.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-02,100,101,99,100,1000\n"
        "2024-01-03,101,102,100,101,1000\n"
        "2024-01-04,102,103,101,102,1000\n"
        "2024-01-05,103,104,102,103,1000\n"
        "2024-01-08,104,105,103,104,1000\n"
    )
    out = run_single_stock_backtest(csv_path=csv, symbol="TEST", strategy="mom20", one_way_bps=1.0)
    assert out.symbol == "TEST"
    assert "equity" in out.frame.columns
