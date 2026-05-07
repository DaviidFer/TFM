"""
MVP histórico de la Fase 10 del TFM (live D1 con MT5).

Combina `SimulationRuntime` (`app/orchestrator/`) con `LiveTradingRuntime` para
validar el ciclo live mínimo. **No participa en el flujo productivo**: el runtime real
es `DevelopmentOperationalSupervisor`. Se conserva como narrativa académica
del cierre de la Fase 10.
"""
from __future__ import annotations

import os
from pathlib import Path
from queue import Queue

from app.agents import AgentContext, DeveloperAgent, PortfolioManagerProcess, TraderAgent, ValidationAgent
from app.cloud import LOCAL_PATHS
from app.core.structured_logging import LOG_FILE_PATH, emit_log
from app.execution.local_data_provider import LocalMarketDataProvider
from app.execution.models import ExecutionMode
from app.execution.mt5_data_provider import MT5DataProvider
from app.execution.mt5_connector import MT5Connector
from app.execution.router import ExecutionRouter
from app.orchestrator.simulation import SimulationRuntime
from app.runtime import LiveTradingRuntime
from app.services import DataProcess
from app.storage import StateStore


def _read_mode() -> ExecutionMode:
    raw = str(os.getenv("EXECUTION_MODE", "paper")).strip().lower()
    return ExecutionMode.LIVE_MT5 if raw == "live_mt5" else ExecutionMode.PAPER


def main(*, db_path: Path | None = None) -> int:
    print("=== Phase 10 Live MVP ===")
    db_path = db_path or LOCAL_PATHS.phase_db(10)
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
    mode = _read_mode()
    mt5 = MT5Connector(env_path=".env")
    # El data provider operativo siempre debe leer desde API MT5.
    if not mt5.connect():
        raise RuntimeError("No se pudo conectar a MT5. Revisa .env, terminal MT5 y credenciales.")

    execution_router = ExecutionRouter(
        market_data=LocalMarketDataProvider(asset_csv_by_symbol=universe),
        mode=mode,
        mt5_connector=mt5,
    )
    ctx = AgentContext(
        store=StateStore(db_path=db_path),
        artifacts_root=LOCAL_PATHS.phase_dir(10),
        execution_router=execution_router,
    )
    data_process = DataProcess(ctx)
    developer_agent = DeveloperAgent(ctx)
    validation_agent = ValidationAgent(ctx)
    trader_agent = TraderAgent(ctx)
    # Portfolio/Risk quedan temporalmente fuera de esta fase live MVP.
    portfolio_manager_process = PortfolioManagerProcess(ctx)
    simulation = SimulationRuntime(
        data_process=data_process,
        developer_agent=developer_agent,
        validation_agent=validation_agent,
        trader_agent=trader_agent,
        portfolio_manager_process=portfolio_manager_process,
    )

    emit_log(
        "phase10_live_mvp",
        "run_started",
        db_path=str(db_path),
        execution_mode=mode.value,
        universe=list(universe.keys()),
        portfolio_enabled=False,
        risk_enabled=False,
    )

    built = simulation.build_candidate_pool(asset_csv_by_asset=universe, timeframe="D1")
    # Compatibilidad defensiva: si hay sesión con módulo antiguo en memoria.
    if hasattr(simulation, "get_promoted_registry"):
        registry = simulation.get_promoted_registry()
    else:
        registry = dict(getattr(simulation, "_promoted_registry", {}))
    if not built or not registry:
        raise RuntimeError("No se generaron traders promovidos.")

    # Activación directa de todos los traders promovidos (sin portfolio/risk en esta fase).
    for spec in registry.values():
        trader_agent.activate(spec)
    emit_log("phase10_live_mvp", "traders_activated", count=len(registry), trader_ids=list(registry.keys()))

    symbols = sorted({spec.asset for spec in registry.values()})
    events_queue: Queue = Queue()
    runtime_data_provider = MT5DataProvider(events_queue=events_queue, symbol_list=symbols, timeframe="1d")

    runtime = LiveTradingRuntime(
        trader_agent=trader_agent,
        promoted_specs=registry,
        data_provider=runtime_data_provider,
        timeframe="1d",
        bars_lookback=260,
    )
    runtime.run(max_cycles=500, idle_sleep_sec=300)

    emit_log("phase10_live_mvp", "run_completed", execution_mode=mode.value, log_file=str(LOG_FILE_PATH))
    print("Phase 10 live MVP started and finished configured loop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

