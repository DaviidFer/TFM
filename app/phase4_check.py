from __future__ import annotations

from pathlib import Path

from app.agents import AgentContext, DataAgent, DeveloperAgent, TraderAgent, ValidationAgent
from app.contracts import TraderLifecycleState
from app.storage import StateStore


def main() -> int:
    print("=== Phase 4 Check ===")
    db_path = Path("app/.tmp/phase4/phase4.sqlite")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    ctx = AgentContext(
        store=StateStore(db_path=db_path),
        artifacts_root=Path("app/.tmp/phase4"),
    )

    data_agent = DataAgent(ctx)
    developer_agent = DeveloperAgent(ctx)
    validation_agent = ValidationAgent(ctx)
    trader_agent = TraderAgent(ctx)

    dataset = data_agent.prepare_dataset(
        asset="AAPL",
        timeframe="D1",
        asset_csv_path="datos/Stocks/AAPL.csv",
    )
    print(f"dataset_ready: {dataset.dataset_id} rows={dataset.rows}")

    dev = developer_agent.develop(
        dataset=dataset,
        families=("decision_tree", "rulefit", "genetico", "quantile", "subgroup"),
        family_params={
            "decision_tree": {"target_n_rules": 60, "progress_every": 0},
            "rulefit": {"target_n_rules": 60, "n_estimators": 35, "max_candidate_rules": 250, "progress_every": 0},
            "genetico": {"target_n_rules": 60, "population_size": 50, "n_generations": 12, "progress_every": 0},
            "quantile": {"n_bins": 4, "combo_size": 1, "min_coverage": 180},
            "subgroup": {"n_bins": 5, "min_coverage": 80},
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

