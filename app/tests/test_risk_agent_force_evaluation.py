from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from app.agents import AgentContext, RiskAgent
from app.contracts import DesignRiskProfile, PromotedTraderSpec, RiskLimitsConfig, TraderForwardMetrics, TraderLifecycleState
from app.runtime.development_operational_supervisor import DevelopmentOperationalSupervisor
from app.storage import StateStore


def _tmp_db_path() -> Path:
    base = Path("app/.tmp/tests")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"risk_force_{uuid4().hex[:8]}.sqlite"


def test_risk_agent_force_evaluation_persists_snapshots(monkeypatch) -> None:
    store = StateStore(db_path=_tmp_db_path())
    ctx = AgentContext(store=store, artifacts_root=Path("app/.tmp/tests"))
    agent = RiskAgent(ctx)
    spec_a = PromotedTraderSpec(trader_id="tr_A", asset="AAPL", timeframe="D1", long_rules=["r1"], short_rules=[], origin_experiment_id="exp1")
    spec_b = PromotedTraderSpec(trader_id="tr_B", asset="MSFT", timeframe="D1", long_rules=["r2"], short_rules=[], origin_experiment_id="exp2")
    store.upsert_trader_state(trader_id="tr_A", asset="AAPL", timeframe="D1", state=TraderLifecycleState.LIVE)
    store.upsert_trader_state(trader_id="tr_B", asset="MSFT", timeframe="D1", state=TraderLifecycleState.LIVE)
    monkeypatch.setattr(agent, "_load_promoted_specs", lambda: {"tr_A": spec_a, "tr_B": spec_b})

    def _profile(spec):
        return DesignRiskProfile(
            trader_id=spec.trader_id,
            asset=spec.asset,
            timeframe=spec.timeframe,
            promoted_at="2026-01-01T00:00:00+00:00",
            sharpe_design=1.2,
            profit_factor_design=1.8,
            max_drawdown_design=0.10,
            avg_loss_design=-10.0,
            winrate_design=0.55,
            expectancy_design=3.0,
            max_losing_streak_design=3,
            trades_design=50,
        )

    monkeypatch.setattr(agent, "_build_design_profile", _profile)

    def _forward(**kwargs):
        trader_id = kwargs["trader_id"]
        if trader_id == "tr_A":
            return TraderForwardMetrics(
                trader_id=trader_id,
                asset="AAPL",
                timeframe="D1",
                evaluation_run_id=str(kwargs["evaluation_run_id"]),
                promoted_at="2026-01-01T00:00:00+00:00",
                evaluation_date="2026-04-01T00:00:00+00:00",
                forward_start="2026-01-01",
                forward_end="2026-04-01",
                shadow_trades=18,
                shadow_sharpe=1.0,
                shadow_profit_factor=1.4,
                shadow_max_drawdown=0.11,
                shadow_avg_loss=-11.0,
                shadow_winrate=0.50,
                shadow_expectancy=1.5,
                shadow_losing_streak=3,
                signal_count=18,
                ppo_selected_count=10,
            )
        return TraderForwardMetrics(
            trader_id=trader_id,
            asset="MSFT",
            timeframe="D1",
            evaluation_run_id=str(kwargs["evaluation_run_id"]),
            promoted_at="2026-01-01T00:00:00+00:00",
            evaluation_date="2026-04-01T00:00:00+00:00",
            forward_start="2026-01-01",
            forward_end="2026-04-01",
            shadow_trades=30,
            shadow_sharpe=-0.3,
            shadow_profit_factor=0.4,
            shadow_max_drawdown=0.28,
            shadow_avg_loss=-22.0,
            shadow_winrate=0.25,
            shadow_expectancy=-5.0,
            shadow_losing_streak=8,
            signal_count=30,
            ppo_selected_count=0,
            ppo_blocked_count=6,
        )

    monkeypatch.setattr(agent.forward_service, "run_forward_backtest_for_trader", _forward)
    snapshots = agent.force_risk_evaluation()
    assert len(snapshots) == 2
    runs = store.list_risk_evaluation_runs()
    details = store.list_risk_evaluation_details()
    assert runs[0]["evaluated_traders"] == 2
    assert len(details) == 2
    assert any(row["action"] == "retraining" for row in details)


def test_supervisor_force_risk_evaluation_updates_status(monkeypatch) -> None:
    supervisor = DevelopmentOperationalSupervisor(db_path=_tmp_db_path())
    monkeypatch.setattr(
        supervisor.risk_agent,
        "evaluate_trader_universe",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        supervisor.ctx.store,
        "list_risk_evaluation_runs",
        lambda limit=1: [{"run_id": "risk_test", "status": "completed", "evaluated_traders": 0, "started_at": "2026-04-01T00:00:00+00:00"}],
    )
    out = supervisor.force_risk_evaluation(force_backtest=False)
    status = supervisor.get_status()
    assert out["status"] == "completed"
    assert status["risk_last_force_evaluation_at"] is not None
    assert status["risk_last_evaluation_run_id"] == "risk_test"
