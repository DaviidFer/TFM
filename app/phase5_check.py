from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.agents import (
    AgentContext,
    DeveloperAgent,
    PortfolioManagerAgent,
    RiskAgent,
    RiskThresholds,
    TraderAgent,
    ValidationAgent,
)
from app.contracts import TraderLiveMetrics, TraderLifecycleState
from app.core.structured_logging import LOG_FILE_PATH, emit_log
from app.orchestrator import RuntimeOrchestrator
from app.services import DataProcess
from app.storage import StateStore


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(*, db_path: Path | None = None) -> int:
    print("=== Phase 5 Check ===")
    db_path = db_path or Path("app/.tmp/phase5/phase5.sqlite")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    if LOG_FILE_PATH.exists():
        LOG_FILE_PATH.unlink()

    ctx = AgentContext(
        store=StateStore(db_path=db_path),
        artifacts_root=Path("app/.tmp/phase5"),
    )
    emit_log("phase5_check", "run_started", run_id=uuid4().hex[:8], db_path=str(db_path), log_file=str(LOG_FILE_PATH))

    data_process = DataProcess(ctx)
    developer_agent = DeveloperAgent(ctx)
    validation_agent = ValidationAgent(ctx)
    trader_agent = TraderAgent(ctx)
    risk_agent = RiskAgent(ctx, thresholds=RiskThresholds(max_drawdown=0.20, min_sharpe=-0.8, min_trades=5))
    portfolio_agent = PortfolioManagerAgent(ctx)
    orchestrator = RuntimeOrchestrator(
        data_process=data_process,
        developer_agent=developer_agent,
        validation_agent=validation_agent,
        trader_agent=trader_agent,
    )

    dataset = data_process.prepare_dataset(asset="AAPL", timeframe="D1", asset_csv_path="datos/Stocks/AAPL.csv")
    dev = developer_agent.develop(
        dataset=dataset,
        families=("decision_tree", "rulefit", "genetico", "quantile"),
        family_params={
            "decision_tree": {"target_n_rules": 40, "progress_every": 0},
            "rulefit": {"target_n_rules": 40, "n_estimators": 30, "max_candidate_rules": 220, "progress_every": 0},
            "genetico": {"target_n_rules": 40, "population_size": 40, "n_generations": 10, "progress_every": 0},
            "quantile": {"n_bins": 4, "combo_size": 2, "min_coverage": 100},
        },
    )
    val = validation_agent.validate_and_promote(dev)
    live = trader_agent.activate(val.promoted_spec)
    print(f"initial_trader_live: {live.trader_id}")
    emit_log(
        "phase5_check",
        "first_cycle_completed",
        dataset_id=dataset.dataset_id,
        experiment_id=dev.experiment_config.experiment_id,
        trader_id=live.trader_id,
    )

    # Publicar métricas degradadas para forzar acción de riesgo.
    degraded = TraderLiveMetrics(
        trader_id=live.trader_id,
        as_of=utc_now_iso(),
        pnl=-2500.0,
        sharpe_rolling=-1.2,
        drawdown_rolling=0.28,
        trade_count=18,
        extra_metrics={"corr_penalty": 0.15},
    )
    trader_agent.publish_metrics(degraded, correlation_id=val.promoted_spec.origin_experiment_id)
    decision = risk_agent.assess_trader(
        trader_id=live.trader_id,
        asset=val.promoted_spec.asset,
        timeframe=val.promoted_spec.timeframe,
    )
    if decision is None:
        raise RuntimeError("RiskAgent did not produce a decision.")
    print(f"risk_decision: action={decision.action} reason={decision.reason}")
    if decision.action != "retraining":
        raise RuntimeError("Expected retraining action in phase5 scenario.")

    # El orquestador consume retrain_requested y crea nuevo ciclo.
    processed = orchestrator.process_pending_retrain_events(
        asset_csv_by_asset={"AAPL": "datos/Stocks/AAPL.csv"},
        families=("decision_tree", "rulefit", "genetico", "quantile"),
        family_params={
            "decision_tree": {"target_n_rules": 25, "progress_every": 0},
            "rulefit": {"target_n_rules": 25, "n_estimators": 25, "max_candidate_rules": 160, "progress_every": 0},
            "genetico": {"target_n_rules": 25, "population_size": 35, "n_generations": 8, "progress_every": 0},
            "quantile": {"n_bins": 4, "combo_size": 2, "min_coverage": 100},
        },
    )
    if len(processed) == 0:
        raise RuntimeError("Orchestrator did not process retrain requests.")
    print(f"retrain_processed: count={len(processed)} new_trader={processed[0]['trader_id']}")
    emit_log("phase5_check", "retrain_processed", processed=processed)

    # Portfolio v1 sobre métricas live actuales
    pm = portfolio_agent.rebalance(as_of=utc_now_iso(), max_weight=0.7, min_score=-0.5)
    print(f"portfolio_decision: selected={len(pm.selected_traders)} weights={pm.weights}")
    emit_log(
        "phase5_check",
        "portfolio_rebalanced",
        selected_traders=pm.selected_traders,
        weights=pm.weights,
    )

    # Validaciones finales de estado/eventos
    old_state = ctx.store.get_trader_state(live.trader_id)
    if old_state is None or old_state.state != TraderLifecycleState.RETRAINING:
        raise RuntimeError("Original trader should be in RETRAINING state.")

    new_trader_id = processed[0]["trader_id"]
    new_state = ctx.store.get_trader_state(new_trader_id)
    if new_state is None or new_state.state != TraderLifecycleState.LIVE:
        raise RuntimeError("New trader should be in LIVE state after retrain pipeline.")

    all_events = ctx.store.list_events(limit=200)
    print(f"events_total: {len(all_events)}")
    if len(all_events) < 12:
        raise RuntimeError("Expected richer event trail in phase 5.")

    emit_log("phase5_check", "run_completed", events_total=len(all_events), log_file=str(LOG_FILE_PATH))
    print("Phase 5 check completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

