from __future__ import annotations

from collections import Counter
from pathlib import Path

from app.agents import (
    AgentContext,
    DataAgent,
    DeveloperAgent,
    PortfolioManagerAgent,
    RiskAgent,
    TraderAgent,
    ValidationAgent,
)
from app.core.structured_logging import LOG_FILE_PATH, emit_log
from app.execution.local_data_provider import LocalMarketDataProvider
from app.execution.models import ExecutionMode
from app.execution.mt5_connector import MT5Connector
from app.execution.router import ExecutionRouter
from app.orchestrator.simulation import SimulationRuntime
from app.storage import StateStore


def main(*, db_path: Path | None = None) -> int:
    print("=== Phase 8 Check ===")
    db_path = db_path or Path("app/.tmp/phase8/phase8.sqlite")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    if LOG_FILE_PATH.exists():
        LOG_FILE_PATH.unlink()

    universe = {
        "GOOGL": "datos/Stocks/GOOGL.csv",
        "AAPL": "datos/Stocks/AAPL.csv",
        "MSFT": "datos/Stocks/MSFT.csv",
    }
    execution_router = ExecutionRouter(
        market_data=LocalMarketDataProvider(asset_csv_by_symbol=universe),
        mode=ExecutionMode.PAPER,
        mt5_connector=MT5Connector(env_path=".env"),
    )
    ctx = AgentContext(
        store=StateStore(db_path=db_path),
        artifacts_root=Path("app/.tmp/phase8"),
        execution_router=execution_router,
    )
    data_agent = DataAgent(ctx)
    developer_agent = DeveloperAgent(ctx)
    validation_agent = ValidationAgent(ctx)
    trader_agent = TraderAgent(ctx)
    risk_agent = RiskAgent(ctx)
    portfolio_agent = PortfolioManagerAgent(ctx)
    simulation = SimulationRuntime(
        data_agent=data_agent,
        developer_agent=developer_agent,
        validation_agent=validation_agent,
        trader_agent=trader_agent,
        portfolio_agent=portfolio_agent,
    )

    emit_log("phase8_check", "run_started", db_path=str(db_path), universe=list(universe.keys()))

    built = simulation.build_candidate_pool(asset_csv_by_asset=universe, timeframe="D1")
    print(f"promoted_candidates: {len(built)}")
    if len(built) < 3:
        raise RuntimeError("Expected at least 3 promoted candidates in phase8.")

    states_before = ctx.store.list_trader_states()
    counts_before = Counter([s.state.value for s in states_before])
    print(f"states_before_activation: {dict(counts_before)}")
    if counts_before.get("promoted", 0) < 3:
        raise RuntimeError("Expected promoted queue before activation.")
    if counts_before.get("live", 0) != 0:
        raise RuntimeError("Expected zero live traders before portfolio activation.")

    activated = simulation.activate_top_candidates(max_live_traders=2, max_weight=0.7, min_score=-0.25)
    print(f"activated_traders: {activated}")
    if len(activated) <= 0:
        raise RuntimeError("Portfolio activation selected no traders.")

    # Integración framework de ejecución: sólo trader/risk/portfolio acceden al router.
    for trader_id in activated:
        state = ctx.store.get_trader_state(trader_id)
        if state is None:
            continue
        route_res = trader_agent.route_order(
            trader_id=trader_id,
            symbol=state.asset,
            side="buy",
            volume=0.1,
            comment="phase8_paper_order",
        )
        if not bool(route_res.get("accepted")):
            raise RuntimeError(f"Expected paper routing accepted for {trader_id}.")

    broker_positions_pm = portfolio_agent.get_broker_positions()
    broker_account_risk = risk_agent.get_broker_account_info()
    emit_log(
        "phase8_check",
        "execution_bridge_snapshot",
        broker_positions_count=len(broker_positions_pm),
        broker_account=broker_account_risk,
    )

    states_after = ctx.store.list_trader_states()
    counts_after = Counter([s.state.value for s in states_after])
    print(f"states_after_activation: {dict(counts_after)}")
    if counts_after.get("live", 0) <= 0:
        raise RuntimeError("Expected at least one LIVE trader after activation.")
    if counts_after.get("promoted", 0) <= 0:
        raise RuntimeError("Expected at least one trader still queued as PROMOTED.")

    metrics = ctx.store.list_trader_metrics()
    print(f"metrics_total: {len(metrics)}")
    if len(metrics) < 3:
        raise RuntimeError("Expected scouting/live metrics for candidate universe.")

    events = ctx.store.list_events(limit=500)
    print(f"events_total: {len(events)}")
    if len(events) < 15:
        raise RuntimeError("Expected rich event trail in phase8 scenario.")

    emit_log(
        "phase8_check",
        "run_completed",
        built_candidates=len(built),
        activated=activated,
        states_after=dict(counts_after),
        events_total=len(events),
        log_file=str(LOG_FILE_PATH),
    )
    print("Phase 8 check completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

