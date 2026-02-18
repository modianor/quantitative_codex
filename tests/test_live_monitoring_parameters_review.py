import pandas as pd

from quantitative_codex.execution.brokers.paper import PaperBrokerAdapter
from quantitative_codex.execution.oms import OMS
from quantitative_codex.live import LiveTradingConfig, SmallCapitalLiveRunner
from quantitative_codex.monitoring import AlertRuleSet, evaluate_alerts
from quantitative_codex.parameters import ParameterRegistry
from quantitative_codex.review import build_postmortem_report


def test_small_capital_runner_generates_budgeted_targets():
    broker = PaperBrokerAdapter()
    oms = OMS(broker)
    runner = SmallCapitalLiveRunner(
        oms,
        LiveTradingConfig(starting_equity=1000, max_notional_per_order=100, max_daily_turnover_ratio=0.1),
    )

    target = pd.Series({"AAPL": 0.8})
    prices = pd.Series({"AAPL": 200.0})
    budgeted = runner.rebalance(target, prices, equity=1000)
    assert "AAPL" in budgeted.index
    assert abs(float(budgeted["AAPL"])) <= 0.5


def test_monitoring_and_postmortem_and_parameter_registry(tmp_path):
    equity = pd.Series([1.0, 0.92, 0.90])
    pnl = pd.Series([0.0, -0.04, -0.01])
    order_log = pd.DataFrame([{"status": "rejected"}, {"status": "submitted"}])
    positions_notional = pd.Series({"AAPL": 1_500_000.0})

    alerts = evaluate_alerts(
        equity,
        pnl,
        order_log,
        positions_notional,
        AlertRuleSet(max_drawdown=0.05, max_daily_loss=0.03, max_reject_ratio=0.2, max_position_abs=1_000_000),
    )
    assert len(alerts) >= 2

    reg = ParameterRegistry(tmp_path / "params.json")
    reg.add("trend", "v1", {"fast": 50, "slow": 200}, note="init")
    reg.add("trend", "v2", {"fast": 40, "slow": 180}, note="retune")
    latest = reg.latest("trend")
    assert latest is not None
    assert latest.version == "v2"

    trades = pd.DataFrame(
        [
            {"symbol": "AAPL", "pnl": 10.0},
            {"symbol": "MSFT", "pnl": -5.0},
        ]
    )
    report = build_postmortem_report(trades, alerts=[{"code": a.code} for a in alerts])
    assert report["summary"]["total_trades"] == 2
    assert "top_symbols" in report
