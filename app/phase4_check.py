"""
Smoke check histórico de la Fase 4 del TFM.

Cierra el pipeline `Data → Developer → Validation → Trader` sobre un activo de
muestra (AAPL D1). **No participa en el flujo productivo**: se conserva como
narrativa académica del cierre de la Fase 4. Se invoca manualmente con
`python -m app.phase4_check`.
"""
from __future__ import annotations

from app.agents import AgentContext, DeveloperAgent, TraderAgent, ValidationAgent
from app.cloud import LOCAL_PATHS
from app.contracts import TraderLifecycleState
from app.services import DataProcess
from app.storage import StateStore


def main() -> int:
    print("=== Phase 4 Check ===")
    db_path = LOCAL_PATHS.phase_db(4)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    ctx = AgentContext(
        store=StateStore(db_path=db_path),
        artifacts_root=LOCAL_PATHS.phase_dir(4),
    )

    data_process = DataProcess(ctx)
    developer_agent = DeveloperAgent(ctx)
    validation_agent = ValidationAgent(ctx)
    trader_agent = TraderAgent(ctx)

    dataset = data_process.prepare_dataset(
        asset="AAPL",
        timeframe="D1",
        asset_csv_path="datos/Stocks/AAPL.csv",
    )
    print(f"dataset_ready: {dataset.dataset_id} rows={dataset.rows}")

    dev = developer_agent.develop(
        dataset=dataset,
        families=("decision_tree", "rulefit", "genetico", "quantile"),
        family_params={
            "decision_tree": {"target_n_rules": 60, "progress_every": 0},
            "rulefit": {"target_n_rules": 60, "n_estimators": 35, "max_candidate_rules": 250, "progress_every": 0},
            "genetico": {"target_n_rules": 60, "population_size": 50, "n_generations": 12, "progress_every": 0},
            "quantile": {"n_bins": 4, "combo_size": 2, "min_coverage": 100},
        },
    )
    print(
        f"candidates_ready: exp={dev.experiment_config.experiment_id} "
        f"long={len(dev.candidate_rules.long_rules)} short={len(dev.candidate_rules.short_rules)}"
    )
    if len(dev.candidate_rules.long_rules) + len(dev.candidate_rules.short_rules) == 0:
        raise RuntimeError("DeveloperAgent did not generate candidate rules.")

    val = validation_agent.validate_and_promote(dev)
    print(
        f"validation_done: stable_long={val.report.passed_long} "
        f"stable_short={val.report.passed_short} trader={val.promoted_spec.trader_id}"
    )
    if len(val.promoted_spec.long_rules) + len(val.promoted_spec.short_rules) == 0:
        raise RuntimeError("ValidationAgent produced empty promoted spec.")

    metrics = trader_agent.activate(val.promoted_spec)
    print(f"trader_live: {metrics.trader_id} readiness={metrics.extra_metrics.get('readiness_score')}")

    row = ctx.store.get_trader_state(val.promoted_spec.trader_id)
    if row is None or row.state != TraderLifecycleState.LIVE:
        raise RuntimeError("TraderAgent did not set LIVE state.")

    events = ctx.store.list_events(limit=20)
    print(f"events_stored: {len(events)}")
    if len(events) < 4:
        raise RuntimeError("Expected at least 4 events in phase 4 flow.")

    print("Phase 4 check completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

