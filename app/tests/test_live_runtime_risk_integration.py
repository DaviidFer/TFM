from __future__ import annotations

from queue import Queue
from types import SimpleNamespace
from pathlib import Path
from uuid import uuid4

from app.agents import AgentContext, RiskAgent, TraderAgent
from app.contracts import PortfolioDecision, PromotedTraderSpec, RiskAdjustedPortfolioDecision
from app.runtime.live_trading_runtime import LiveTradingRuntime
from app.storage import StateStore


class _DummyProvider:
    def __init__(self) -> None:
        self.events_queue = Queue()
        self.timeframe = "1d"


def _tmp_db_path() -> Path:
    base = Path("app/.tmp/tests")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"live_risk_{uuid4().hex[:8]}.sqlite"


def test_live_runtime_blocks_order_for_trader_rejected_by_risk(monkeypatch) -> None:
    store = StateStore(db_path=_tmp_db_path())
    ctx = AgentContext(store=store, artifacts_root=Path("app/.tmp/tests"))
    trader_agent = TraderAgent(ctx)
    risk_agent = RiskAgent(ctx)
    provider = _DummyProvider()
    spec_a = PromotedTraderSpec(trader_id="tr_A", asset="AAPL", timeframe="D1", long_rules=["r1"], short_rules=[], origin_experiment_id="exp1")
    spec_b = PromotedTraderSpec(trader_id="tr_B", asset="MSFT", timeframe="D1", long_rules=["r2"], short_rules=[], origin_experiment_id="exp2")
    runtime = LiveTradingRuntime(
        trader_agent=trader_agent,
        risk_agent=risk_agent,
        portfolio_manager=None,
        promoted_specs={"tr_A": spec_a, "tr_B": spec_b},
        data_provider=provider,
    )
    calls: list[dict] = []

    def _fake_route_order(**kwargs):
        calls.append(dict(kwargs))
        return {"accepted": True, "reason": "", "ticket": 1}

    monkeypatch.setattr(trader_agent, "route_order", _fake_route_order)
    monkeypatch.setattr(
        risk_agent,
        "review_portfolio_decision",
        lambda **kwargs: RiskAdjustedPortfolioDecision(
            rebalance_id="rb_1",
            evaluation_id="rpc_1",
            original_decision=kwargs["portfolio_decision"],
            approved=True,
            action="approve_with_clipping",
            adjusted_weights={"tr_A": 0.5},
            original_weights={"tr_A": 0.5, "tr_B": 0.5},
            forced_cash_weight=0.5,
            blocked_traders=["tr_B"],
            clipped_traders=[],
            reasons=["tr_B blocked"],
            diagnostics={},
        ),
    )
    runtime._process_signal_candidates(
        [
            {"trader_id": "tr_A", "symbol": "AAPL", "side": "buy", "signal_label": "SignalType.BUY", "price": 100.0, "spec": spec_a, "detected_at": "2026-04-01T00:00:00+00:00"},
            {"trader_id": "tr_B", "symbol": "MSFT", "side": "buy", "signal_label": "SignalType.BUY", "price": 100.0, "spec": spec_b, "detected_at": "2026-04-01T00:00:00+00:00"},
        ],
        force_rebalance=True,
    )
    assert len(calls) == 1
    assert calls[0]["trader_id"] == "tr_A"
