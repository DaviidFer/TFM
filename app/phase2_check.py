from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.contracts import (
    AgentKind,
    CandidateRules,
    DatasetContract,
    EventType,
    ExperimentConfig,
    PromotedTraderSpec,
    RetrainRequest,
    TraderLifecycleState,
    ValidationReport,
)
from app.storage import StateStore


def _simulate_activation(store: StateStore, trader_id: str, asset: str, timeframe: str) -> None:
    """Activacion del trader: pasa directo a LIVE (no hay estados intermedios)."""
    store.upsert_trader_state(
        trader_id=trader_id,
        asset=asset,
        timeframe=timeframe,
        state=TraderLifecycleState.LIVE,
        notes="activated by trader agent",
    )
    row = store.get_trader_state(trader_id)
    if row is None or row.state != TraderLifecycleState.LIVE:
        raise RuntimeError("Activation simulation failed: expected LIVE state.")


def _simulate_retraining(store: StateStore, trader_id: str, asset: str, timeframe: str) -> None:
    """Trader que falla validacion de Risk -> RETRAINING (cash) y RetrainRequest."""
    retrain = RetrainRequest(
        request_id=f"rr_{uuid4().hex[:10]}",
        trader_id=trader_id,
        asset=asset,
        timeframe=timeframe,
        reason="drawdown breach",
        requested_by=AgentKind.RISK,
    )
    store.append_event(
        event_id=f"evt_{uuid4().hex[:10]}",
        event_type=EventType.RETRAIN_REQUESTED,
        producer=AgentKind.RISK.value,
        payload=retrain.to_dict(),
    )
    store.upsert_trader_state(
        trader_id=trader_id,
        asset=asset,
        timeframe=timeframe,
        state=TraderLifecycleState.RETRAINING,
        notes="retrain requested",
    )
    row = store.get_trader_state(trader_id)
    if row is None or row.state != TraderLifecycleState.RETRAINING:
        raise RuntimeError("Retraining simulation failed.")


def main() -> int:
    print("=== Phase 2 Check ===")
    dataset = DatasetContract(
        dataset_id="ds_aapl_d1_v1",
        asset="AAPL",
        timeframe="D1",
        source_path="datos/Stocks/AAPL.csv",
        rows=5000,
        start_date="2000-01-01",
        end_date="2026-01-01",
        quality_score=0.98,
    )
    exp = ExperimentConfig(
        experiment_id="exp_aapl_001",
        asset="AAPL",
        timeframe="D1",
        split_policy="is_oos_holdout_2025",
        model_families=["decision_tree", "rulefit", "genetico"],
        parameters={"is_pct": 0.5, "oos_pct": 0.5, "holdout_year": 2025},
    )
    candidates = CandidateRules(
        experiment_id=exp.experiment_id,
        asset="AAPL",
        long_rules=["(RSI_14 > 30) & (RSI_14 <= 55)"],
        short_rules=["(WPR_14 <= -80) & (Stoch_14 <= 20)"],
        generation_summary={"n_long": 1, "n_short": 1},
    )
    report = ValidationReport(
        experiment_id=exp.experiment_id,
        asset="AAPL",
        passed_long=1,
        passed_short=1,
        failed_long=0,
        failed_short=0,
        notes="phase2 smoke",
    )
    promoted = PromotedTraderSpec(
        trader_id="tr_aapl_d1_001",
        asset="AAPL",
        timeframe="D1",
        long_rules=candidates.long_rules,
        short_rules=candidates.short_rules,
        origin_experiment_id=exp.experiment_id,
    )

    # Serializacion minima de contratos
    _ = dataset.to_dict()
    _ = exp.to_dict()
    _ = candidates.to_dict()
    _ = report.to_dict()
    _ = promoted.to_dict()
    print("Contract serialization: OK")

    tmp_dir = Path("app/.tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_dir / "phase2_check.sqlite"
    if db_path.exists():
        db_path.unlink()

    store = StateStore(db_path=db_path)
    _simulate_activation(store, trader_id=promoted.trader_id, asset=promoted.asset, timeframe=promoted.timeframe)
    print("Lifecycle activation -> LIVE: OK")

    _simulate_retraining(store, trader_id=promoted.trader_id, asset=promoted.asset, timeframe=promoted.timeframe)
    print("Lifecycle live -> RETRAINING + event: OK")

    events = store.list_events(limit=10)
    if len(events) == 0:
        raise RuntimeError("Expected at least one event in store.")
    print(f"Stored events: {len(events)}")

    print("Phase 2 check completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

