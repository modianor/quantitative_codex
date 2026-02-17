import pandas as pd

from quantitative_codex.data.providers import load_eod_csv


def test_load_eod_csv_normalizes_and_adds_adj_close(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text(
        "Date,Open,High,Low,Close,Volume,split_factor,div_cash\n"
        "2024-01-02,100,101,99,100,1000,1,0\n"
        "2024-01-03,101,103,100,102,1200,1,0\n"
    )

    df = load_eod_csv(path)
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "split_factor", "div_cash", "adj_close"]
    assert "adj_close" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df.index)
