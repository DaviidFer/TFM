from __future__ import annotations

from pathlib import Path

from app.agents import (
    AgentContext,
    DeveloperAgent,
    PortfolioManagerProcess,
    TraderAgent,
    ValidationAgent,
)
from app.core.structured_logging import LOG_FILE_PATH, emit_log
from app.execution.access import ensure_execution_access
from app.execution.local_data_provider import LocalMarketDataProvider
from app.execution.models import ExecutionMode
from app.execution.mt5_connector import MT5Connector
from app.execution.router import ExecutionRouter
from app.orchestrator.simulation import SimulationRuntime
from app.services import DataProcess
from app.storage import StateStore


def main(*, db_path: Path | None = None) -> int:
    print("=== Phase 9 Check ===")
    db_path = db_path or Path("app/.tmp/phase9/phase9.sqlite")
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
        artifacts_root=Path("app/.tmp/phase9"),
        execution_router=execution_router,
    )
    data_process = DataProcess(ctx)
    developer_agent = DeveloperAgent(ctx)
    validation_agent = ValidationAgent(ctx)
    trader_agent = TraderAgent(ctx)
    portfolio_manager_process = PortfolioManagerProcess(ctx)
    simulation = SimulationRuntime(
        data_process=data_process,
        developer_agent=developer_agent,
        validation_agent=validation_agent,
        trader_agent=trader_agent,
        portfolio_manager_process=portfolio_manager_process,
    )

    emit_log("phase9_check", "run_started", db_path=str(db_path), mode="paper", universe=list(universe.keys()))
    built = simulation.build_candidate_pool(asset_csv_by_asset=universe, timeframe="D1")
    activated = simulation.activate_top_candidates(max_live_traders=2, max_weight=0.7, min_score=-0.25)
    if len(built) < 3:
        raise RuntimeError("Expected at least 3 promoted candidates.")
    if len(activated) < 1:
        raise RuntimeError("Expected at least 1 activated trader.")

    # Trader enruta órdenes (permitido)
    ok = 0
    for tr_id in activated:
        row = ctx.store.get_trader_state(tr_id)
        if row is None:
            continue
        res = trader_agent.route_order(
            trader_id=tr_id,
            symbol=row.asset,
            side="buy",
            volume=0.1,
            comment="phase9_execution_bridge",
        )
        if bool(res.get("accepted")):
            ok += 1
    if ok <= 0:
        raise RuntimeError("Expected routed paper orders from trader.")

    # Portfolio puede leer ejecucion (permitido); Recursos Humanos ya no toca el broker.
    pm_positions = portfolio_manager_process.get_broker_positions()
    print(f"pm_positions: {len(pm_positions)}")

    # Data/Developer/Validation no deben tener acceso
    denied = False
    try:
        ensure_execution_access(data_process.agent_id)
    except PermissionError:
        denied = True
    if not denied:
        raise RuntimeError("Expected execution access denied for data_process.")

    events = ctx.store.list_events(limit=500)
    print(f"events_total: {len(events)}")
    if len(events) < 20:
        raise RuntimeError("Expected rich event trail with execution bridge.")
    emit_log(
        "phase9_check",
        "run_completed",
        built=len(built),
        activated=len(activated),
        routed_orders=ok,
        events_total=len(events),
        log_file=str(LOG_FILE_PATH),
    )
    print("Phase 9 check completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

