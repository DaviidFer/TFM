from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.agents import AgentContext, RiskAgent
from app.contracts import PortfolioDecision, RiskAction, RiskLimitsConfig, TraderLifecycleState
from app.storage import StateStore


def _tmp_db_path() -> Path:
    base = Path("app/.tmp/tests")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"risk_gate_{uuid4().hex[:8]}.sqlite"


def _risk_agent() -> RiskAgent:
    store = StateStore(db_path=_tmp_db_path())
    ctx = AgentContext(store=store, artifacts_root=Path("app/.tmp/tests"))
    store.upsert_trader_state(trader_id="tr_live", asset="AAPL", timeframe="D1", state=TraderLifecycleState.LIVE)
    store.upsert_trader_state(trader_id="tr_live2", asset="MSFT", timeframe="D1", state=TraderLifecycleState.LIVE)
    store.upsert_trader_state(trader_id="tr_retrain", asset="NVDA", timeframe="D1", state=TraderLifecycleState.RETRAINING)
    return RiskAgent(ctx)


def test_decision_ppo_valida_returns_approve() -> None:
    agent = _risk_agent()
    decision = PortfolioDecision(decision_id="rb_1", as_of="2026-04-01T00:00:00+00:00", selected_traders=["tr_live"], weights={"tr_live": 0.20}, target_cash_weight=0.8)
    out = agent.review_portfolio_decision(decision, account_info={"balance": 100000, "equity": 99000}, open_positions=[], limits=RiskLimitsConfig(max_weight_per_trader=0.30, max_weight_per_asset=0.40, min_cash_buffer=0.10))
    assert out.action == RiskAction.APPROVE.value
    assert out.adjusted_weights["tr_live"] == 0.20


def test_retraining_trader_is_blocked_to_cash() -> None:
    agent = _risk_agent()
    decision = PortfolioDecision(decision_id="rb_2", as_of="2026-04-01T00:00:00+00:00", selected_traders=["tr_retrain"], weights={"tr_retrain": 0.25}, target_cash_weight=0.0)
    out = agent.review_portfolio_decision(decision, account_info=None, open_positions=[], limits=RiskLimitsConfig())
    assert "tr_retrain" in out.blocked_traders
    assert out.adjusted_weights == {}
    assert out.forced_cash_weight >= 1.0 - 1e-9


def test_excessive_weight_is_clipped() -> None:
    agent = _risk_agent()
    decision = PortfolioDecision(decision_id="rb_4", as_of="2026-04-01T00:00:00+00:00", selected_traders=["tr_live"], weights={"tr_live": 0.40}, target_cash_weight=0.0)
    out = agent.review_portfolio_decision(decision, account_info=None, open_positions=[], limits=RiskLimitsConfig(max_weight_per_trader=0.15, max_weight_per_asset=0.50, min_cash_buffer=0.05))
    assert out.action == RiskAction.APPROVE_WITH_CLIPPING.value
    assert out.adjusted_weights["tr_live"] == 0.15


def test_total_exposure_and_cash_buffer_scale_down() -> None:
    agent = _risk_agent()
    decision = PortfolioDecision(
        decision_id="rb_5",
        as_of="2026-04-01T00:00:00+00:00",
        selected_traders=["tr_live", "tr_live2"],
        weights={"tr_live": 0.70, "tr_live2": 0.20},
        target_cash_weight=0.10,
    )
    out = agent.review_portfolio_decision(
        decision,
        account_info=None,
        open_positions=[],
        limits=RiskLimitsConfig(max_weight_per_trader=0.80, max_weight_per_asset=0.80, max_total_exposure=1.0, min_cash_buffer=0.20),
    )
    assert out.action in {RiskAction.SCALE_DOWN.value, RiskAction.APPROVE_WITH_CLIPPING.value}
    assert sum(out.adjusted_weights.values()) <= 0.80 + 1e-9
