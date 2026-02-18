import pandas as pd

from quantitative_codex.execution.brokers.paper import PaperBrokerAdapter
from quantitative_codex.execution.oms import OMS


def test_oms_submits_and_updates_positions_from_paper_fills():
    broker = PaperBrokerAdapter()
    oms = OMS(broker)

    target = pd.Series({"AAPL": 10.0, "MSFT": 5.0})
    orders = oms.generate_orders_from_target(target)
    reports = oms.submit_orders(orders)
    assert len(reports) == 2

    broker.process_market_data("AAPL", 100.0)
    broker.process_market_data("MSFT", 200.0)
    oms.sync()

    pos = oms.positions.snapshot()
    assert pos["AAPL"] == 10.0
    assert pos["MSFT"] == 5.0
