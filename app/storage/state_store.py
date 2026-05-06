from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.contracts.enums import AgentStatus, EventType, TraderLifecycleState


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TraderStateRow:
    trader_id: str
    asset: str
    timeframe: str
    state: TraderLifecycleState
    updated_at: str
    notes: str = ""


class StateStore:
    """
    Estado compartido minimo para Fase 2.
    - tabla de estado de traders
    - tabla de estado de agentes
    - tabla de eventos
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trader_states (
                    trader_id TEXT PRIMARY KEY,
                    asset TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    state TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT ''
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_status (
                    agent_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    message TEXT NOT NULL DEFAULT ''
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    producer TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    correlation_id TEXT,
                    payload_json TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trader_metrics_latest (
                    trader_id TEXT PRIMARY KEY,
                    as_of TEXT NOT NULL,
                    pnl REAL NOT NULL,
                    sharpe_rolling REAL NOT NULL,
                    drawdown_rolling REAL NOT NULL,
                    trade_count INTEGER NOT NULL,
                    extra_json TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_orders (
                    pending_key TEXT PRIMARY KEY,
                    trader_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    volume REAL NOT NULL,
                    signal_label TEXT NOT NULL,
                    correlation_id TEXT,
                    attempts INTEGER NOT NULL,
                    next_retry_at TEXT NOT NULL,
                    last_reason TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_universe_registry (
                    trader_id TEXT PRIMARY KEY,
                    asset TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    promotion_date TEXT NOT NULL,
                    lifecycle_state TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_training_runs (
                    run_id TEXT PRIMARY KEY,
                    run_type TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL DEFAULT '',
                    algorithm TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    device TEXT NOT NULL,
                    hyperparameters_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    artifacts_json TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT ''
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_training_metrics (
                    run_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    split TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, step, split, metric_name)
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_model_registry (
                    model_version TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    checkpoint_path TEXT NOT NULL,
                    universe_size INTEGER NOT NULL,
                    trained_at TEXT NOT NULL DEFAULT '',
                    fine_tuned_at TEXT NOT NULL DEFAULT '',
                    config_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_rebalance_snapshots (
                    rebalance_id TEXT PRIMARY KEY,
                    rebalance_date TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    training_run_id TEXT NOT NULL DEFAULT '',
                    fine_tune_run_id TEXT NOT NULL DEFAULT '',
                    active_traders_json TEXT NOT NULL,
                    selected_traders_json TEXT NOT NULL,
                    target_weights_json TEXT NOT NULL,
                    target_cash_weight REAL NOT NULL,
                    diagnostics_json TEXT NOT NULL,
                    forward_metrics_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_forward_evaluations (
                    evaluation_id TEXT PRIMARY KEY,
                    rebalance_id TEXT NOT NULL,
                    benchmark_name TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    cumulative_return_1y REAL NOT NULL,
                    sharpe_1y REAL NOT NULL,
                    max_drawdown_1y REAL NOT NULL,
                    curve_points_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trader_backtest_runs (
                    run_id TEXT PRIMARY KEY,
                    trader_id TEXT NOT NULL,
                    asset TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    cutoff_date TEXT NOT NULL,
                    rules_hash TEXT NOT NULL,
                    price_data_fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trader_backtest_artifacts (
                    run_id TEXT PRIMARY KEY,
                    trader_id TEXT NOT NULL,
                    historical_trades_path TEXT NOT NULL,
                    historical_pnl_path TEXT NOT NULL,
                    weekly_signal_mask_path TEXT NOT NULL,
                    weekly_returns_path TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trader_weekly_signal_mask (
                    run_id TEXT NOT NULL,
                    trader_id TEXT NOT NULL,
                    week_end TEXT NOT NULL,
                    active INTEGER NOT NULL,
                    side TEXT NOT NULL DEFAULT '',
                    bars_in_market INTEGER NOT NULL DEFAULT 0,
                    pnl_week REAL NOT NULL DEFAULT 0.0,
                    mask_source TEXT NOT NULL DEFAULT 'real_backtest',
                    PRIMARY KEY (run_id, trader_id, week_end)
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trader_weekly_returns (
                    run_id TEXT NOT NULL,
                    trader_id TEXT NOT NULL,
                    week_end TEXT NOT NULL,
                    weekly_return REAL NOT NULL,
                    equity_close REAL NOT NULL DEFAULT 0.0,
                    balance_close REAL NOT NULL DEFAULT 0.0,
                    PRIMARY KEY (run_id, trader_id, week_end)
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trader_design_profiles (
                    trader_id TEXT PRIMARY KEY,
                    asset TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    promoted_at TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_evaluation_runs (
                    run_id TEXT PRIMARY KEY,
                    run_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL DEFAULT '',
                    evaluated_traders INTEGER NOT NULL DEFAULT 0,
                    retraining_count INTEGER NOT NULL DEFAULT 0,
                    retrain_requests_count INTEGER NOT NULL DEFAULT 0,
                    notes TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL
                );
                """
            )
            # Migracion no destructiva: BBDD existentes con la version anterior
            # (columnas degraded_count/suspended_count/retired_count) no tienen
            # la columna retraining_count. La anadimos si falta.
            try:
                conn.execute(
                    "ALTER TABLE risk_evaluation_runs ADD COLUMN retraining_count INTEGER NOT NULL DEFAULT 0"
                )
            except sqlite3.OperationalError:
                pass
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trader_forward_backtest_runs (
                    run_id TEXT PRIMARY KEY,
                    evaluation_run_id TEXT NOT NULL,
                    trader_id TEXT NOT NULL,
                    asset TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    promoted_at TEXT NOT NULL,
                    forward_start TEXT NOT NULL,
                    forward_end TEXT NOT NULL,
                    status TEXT NOT NULL,
                    artifact_paths_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trader_forward_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluation_run_id TEXT NOT NULL,
                    trader_id TEXT NOT NULL,
                    asset TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    evaluation_date TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_evaluation_details (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluation_run_id TEXT NOT NULL,
                    trader_id TEXT NOT NULL,
                    asset TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    previous_state TEXT NOT NULL,
                    new_state TEXT NOT NULL,
                    action TEXT NOT NULL,
                    health_score REAL NOT NULL,
                    reasons_json TEXT NOT NULL,
                    flags_json TEXT NOT NULL,
                    retrain_request_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS retrain_requests (
                    request_id TEXT PRIMARY KEY,
                    trader_id TEXT NOT NULL,
                    asset TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    consumed_at TEXT NOT NULL DEFAULT '',
                    completed_at TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_portfolio_checks (
                    check_id TEXT PRIMARY KEY,
                    rebalance_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    approved INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    original_weights_json TEXT NOT NULL,
                    adjusted_weights_json TEXT NOT NULL,
                    blocked_traders_json TEXT NOT NULL,
                    clipped_traders_json TEXT NOT NULL,
                    reasons_json TEXT NOT NULL,
                    diagnostics_json TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trader_signal_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    trader_id TEXT NOT NULL,
                    asset TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    signal_side TEXT NOT NULL,
                    signal_active INTEGER NOT NULL,
                    ppo_selected INTEGER NOT NULL,
                    ppo_weight REAL NOT NULL,
                    risk_approved INTEGER NOT NULL,
                    executed INTEGER NOT NULL,
                    hypothetical_return REAL NOT NULL DEFAULT 0.0,
                    executed_return REAL NOT NULL DEFAULT 0.0,
                    reason_if_blocked TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL
                );
                """
            )
            conn.commit()

    def upsert_trader_state(
        self,
        trader_id: str,
        asset: str,
        timeframe: str,
        state: TraderLifecycleState,
        notes: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trader_states (trader_id, asset, timeframe, state, updated_at, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(trader_id) DO UPDATE SET
                    asset=excluded.asset,
                    timeframe=excluded.timeframe,
                    state=excluded.state,
                    updated_at=excluded.updated_at,
                    notes=excluded.notes;
                """,
                (trader_id, asset, timeframe, state.value, utc_now_iso(), notes),
            )
            conn.commit()

    def get_trader_state(self, trader_id: str) -> Optional[TraderStateRow]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT trader_id, asset, timeframe, state, updated_at, notes FROM trader_states WHERE trader_id=?",
                (trader_id,),
            ).fetchone()
            if row is None:
                return None
            return TraderStateRow(
                trader_id=row["trader_id"],
                asset=row["asset"],
                timeframe=row["timeframe"],
                state=TraderLifecycleState(row["state"]),
                updated_at=row["updated_at"],
                notes=row["notes"],
            )

    def list_trader_states(self) -> List[TraderStateRow]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT trader_id, asset, timeframe, state, updated_at, notes FROM trader_states ORDER BY updated_at DESC"
            ).fetchall()
            return [
                TraderStateRow(
                    trader_id=r["trader_id"],
                    asset=r["asset"],
                    timeframe=r["timeframe"],
                    state=TraderLifecycleState(r["state"]),
                    updated_at=r["updated_at"],
                    notes=r["notes"],
                )
                for r in rows
            ]

    def set_agent_status(self, agent_id: str, status: AgentStatus, message: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_status (agent_id, status, updated_at, message)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(agent_id) DO UPDATE SET
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    message=excluded.message;
                """,
                (agent_id, status.value, utc_now_iso(), message),
            )
            conn.commit()

    def list_agent_status(self) -> List[Dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT agent_id, status, updated_at, message FROM agent_status ORDER BY updated_at DESC"
            ).fetchall()
            return [
                {
                    "agent_id": r["agent_id"],
                    "status": r["status"],
                    "updated_at": r["updated_at"],
                    "message": r["message"],
                }
                for r in rows
            ]

    def append_event(
        self,
        event_id: str,
        event_type: EventType,
        producer: str,
        payload: Dict[str, Any],
        occurred_at: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events (event_id, event_type, producer, occurred_at, correlation_id, payload_json)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    event_id,
                    event_type.value,
                    producer,
                    occurred_at or utc_now_iso(),
                    correlation_id,
                    json.dumps(payload, ensure_ascii=True),
                ),
            )
            conn.commit()

    def upsert_trader_metrics(
        self,
        *,
        trader_id: str,
        as_of: str,
        pnl: float,
        sharpe_rolling: float,
        drawdown_rolling: float,
        trade_count: int,
        extra: Dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trader_metrics_latest
                    (trader_id, as_of, pnl, sharpe_rolling, drawdown_rolling, trade_count, extra_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trader_id) DO UPDATE SET
                    as_of=excluded.as_of,
                    pnl=excluded.pnl,
                    sharpe_rolling=excluded.sharpe_rolling,
                    drawdown_rolling=excluded.drawdown_rolling,
                    trade_count=excluded.trade_count,
                    extra_json=excluded.extra_json;
                """,
                (
                    trader_id,
                    as_of,
                    float(pnl),
                    float(sharpe_rolling),
                    float(drawdown_rolling),
                    int(trade_count),
                    json.dumps(extra, ensure_ascii=True),
                ),
            )
            conn.commit()

    def get_trader_metrics(self, trader_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT trader_id, as_of, pnl, sharpe_rolling, drawdown_rolling, trade_count, extra_json
                FROM trader_metrics_latest
                WHERE trader_id=?
                """,
                (trader_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "trader_id": row["trader_id"],
                "as_of": row["as_of"],
                "pnl": float(row["pnl"]),
                "sharpe_rolling": float(row["sharpe_rolling"]),
                "drawdown_rolling": float(row["drawdown_rolling"]),
                "trade_count": int(row["trade_count"]),
                "extra_metrics": json.loads(row["extra_json"]),
            }

    def list_trader_metrics(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT trader_id, as_of, pnl, sharpe_rolling, drawdown_rolling, trade_count, extra_json
                FROM trader_metrics_latest
                ORDER BY as_of DESC
                """
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for row in rows:
                out.append(
                    {
                        "trader_id": row["trader_id"],
                        "as_of": row["as_of"],
                        "pnl": float(row["pnl"]),
                        "sharpe_rolling": float(row["sharpe_rolling"]),
                        "drawdown_rolling": float(row["drawdown_rolling"]),
                        "trade_count": int(row["trade_count"]),
                        "extra_metrics": json.loads(row["extra_json"]),
                    }
                )
            return out

    def list_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT event_id, event_type, producer, occurred_at, correlation_id, payload_json
                FROM events
                ORDER BY occurred_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                out.append(
                    {
                        "event_id": r["event_id"],
                        "event_type": r["event_type"],
                        "producer": r["producer"],
                        "occurred_at": r["occurred_at"],
                        "correlation_id": r["correlation_id"],
                        "payload": json.loads(r["payload_json"]),
                    }
                )
            return out

    def list_events_by_type(self, event_type: str, limit: int = 10000) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT event_id, event_type, producer, occurred_at, correlation_id, payload_json
                FROM events
                WHERE event_type = ?
                ORDER BY occurred_at DESC
                LIMIT ?
                """,
                (str(event_type), int(limit)),
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                out.append(
                    {
                        "event_id": r["event_id"],
                        "event_type": r["event_type"],
                        "producer": r["producer"],
                        "occurred_at": r["occurred_at"],
                        "correlation_id": r["correlation_id"],
                        "payload": json.loads(r["payload_json"]),
                    }
                )
            return out

    def upsert_pending_order(
        self,
        *,
        pending_key: str,
        trader_id: str,
        symbol: str,
        side: str,
        volume: float,
        signal_label: str,
        correlation_id: str | None,
        attempts: int,
        next_retry_at: str,
        last_reason: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO pending_orders
                    (pending_key, trader_id, symbol, side, volume, signal_label, correlation_id, attempts, next_retry_at, last_reason, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pending_key) DO UPDATE SET
                    trader_id=excluded.trader_id,
                    symbol=excluded.symbol,
                    side=excluded.side,
                    volume=excluded.volume,
                    signal_label=excluded.signal_label,
                    correlation_id=excluded.correlation_id,
                    attempts=excluded.attempts,
                    next_retry_at=excluded.next_retry_at,
                    last_reason=excluded.last_reason,
                    updated_at=excluded.updated_at;
                """,
                (
                    pending_key,
                    trader_id,
                    symbol,
                    side,
                    float(volume),
                    signal_label,
                    correlation_id,
                    int(attempts),
                    next_retry_at,
                    last_reason,
                    utc_now_iso(),
                ),
            )
            conn.commit()

    def delete_pending_order(self, pending_key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM pending_orders WHERE pending_key = ?;", (pending_key,))
            conn.commit()

    def list_pending_orders(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT pending_key, trader_id, symbol, side, volume, signal_label, correlation_id,
                       attempts, next_retry_at, last_reason, updated_at
                FROM pending_orders
                ORDER BY next_retry_at ASC
                """
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                out.append(
                    {
                        "pending_key": r["pending_key"],
                        "trader_id": r["trader_id"],
                        "symbol": r["symbol"],
                        "side": r["side"],
                        "volume": float(r["volume"]),
                        "signal_label": r["signal_label"],
                        "correlation_id": r["correlation_id"],
                        "attempts": int(r["attempts"]),
                        "next_retry_at": r["next_retry_at"],
                        "last_reason": r["last_reason"],
                        "updated_at": r["updated_at"],
                    }
                )
            return out

    def clear_all(self) -> None:
        """
        Limpia todo el estado compartido sin depender de borrar el fichero SQLite.
        Util en Windows cuando el unlink del archivo falla por locks temporales.
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM trader_states;")
            conn.execute("DELETE FROM agent_status;")
            conn.execute("DELETE FROM events;")
            conn.execute("DELETE FROM trader_metrics_latest;")
            conn.execute("DELETE FROM pending_orders;")
            conn.execute("DELETE FROM portfolio_universe_registry;")
            conn.execute("DELETE FROM portfolio_training_runs;")
            conn.execute("DELETE FROM portfolio_training_metrics;")
            conn.execute("DELETE FROM portfolio_model_registry;")
            conn.execute("DELETE FROM portfolio_rebalance_snapshots;")
            conn.execute("DELETE FROM portfolio_forward_evaluations;")
            conn.execute("DELETE FROM trader_backtest_runs;")
            conn.execute("DELETE FROM trader_backtest_artifacts;")
            conn.execute("DELETE FROM trader_weekly_signal_mask;")
            conn.execute("DELETE FROM trader_weekly_returns;")
            conn.execute("DELETE FROM trader_design_profiles;")
            conn.execute("DELETE FROM risk_evaluation_runs;")
            conn.execute("DELETE FROM trader_forward_backtest_runs;")
            conn.execute("DELETE FROM trader_forward_metrics;")
            conn.execute("DELETE FROM risk_evaluation_details;")
            conn.execute("DELETE FROM retrain_requests;")
            conn.execute("DELETE FROM risk_portfolio_checks;")
            conn.execute("DELETE FROM trader_signal_audit;")
            conn.commit()

    def upsert_trader_backtest_run(
        self,
        *,
        run_id: str,
        trader_id: str,
        asset: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        cutoff_date: str,
        rules_hash: str,
        price_data_fingerprint: str,
        status: str,
        summary: Dict[str, Any] | None = None,
    ) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trader_backtest_runs
                    (run_id, trader_id, asset, timeframe, start_date, end_date, cutoff_date,
                     rules_hash, price_data_fingerprint, status, summary_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    trader_id=excluded.trader_id,
                    asset=excluded.asset,
                    timeframe=excluded.timeframe,
                    start_date=excluded.start_date,
                    end_date=excluded.end_date,
                    cutoff_date=excluded.cutoff_date,
                    rules_hash=excluded.rules_hash,
                    price_data_fingerprint=excluded.price_data_fingerprint,
                    status=excluded.status,
                    summary_json=excluded.summary_json,
                    updated_at=excluded.updated_at;
                """,
                (
                    run_id,
                    trader_id,
                    asset,
                    timeframe,
                    start_date,
                    end_date,
                    cutoff_date,
                    rules_hash,
                    price_data_fingerprint,
                    status,
                    json.dumps(summary or {}, ensure_ascii=True),
                    now,
                    now,
                ),
            )
            conn.commit()

    def get_latest_trader_backtest_run(self, trader_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT run_id, trader_id, asset, timeframe, start_date, end_date, cutoff_date,
                       rules_hash, price_data_fingerprint, status, summary_json, created_at, updated_at
                FROM trader_backtest_runs
                WHERE trader_id = ?
                ORDER BY cutoff_date DESC, updated_at DESC
                LIMIT 1
                """,
                (trader_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "run_id": row["run_id"],
                "trader_id": row["trader_id"],
                "asset": row["asset"],
                "timeframe": row["timeframe"],
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "cutoff_date": row["cutoff_date"],
                "rules_hash": row["rules_hash"],
                "price_data_fingerprint": row["price_data_fingerprint"],
                "status": row["status"],
                "summary": json.loads(row["summary_json"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }

    def list_trader_backtest_runs(self, trader_id: str | None = None, limit: int = 500) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            if trader_id:
                rows = conn.execute(
                    """
                    SELECT run_id, trader_id, asset, timeframe, start_date, end_date, cutoff_date,
                           rules_hash, price_data_fingerprint, status, summary_json, created_at, updated_at
                    FROM trader_backtest_runs
                    WHERE trader_id = ?
                    ORDER BY cutoff_date DESC, updated_at DESC
                    LIMIT ?
                    """,
                    (trader_id, int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT run_id, trader_id, asset, timeframe, start_date, end_date, cutoff_date,
                           rules_hash, price_data_fingerprint, status, summary_json, created_at, updated_at
                    FROM trader_backtest_runs
                    ORDER BY cutoff_date DESC, updated_at DESC
                    LIMIT ?
                    """,
                    (int(limit),),
                ).fetchall()
            return [
                {
                    "run_id": r["run_id"],
                    "trader_id": r["trader_id"],
                    "asset": r["asset"],
                    "timeframe": r["timeframe"],
                    "start_date": r["start_date"],
                    "end_date": r["end_date"],
                    "cutoff_date": r["cutoff_date"],
                    "rules_hash": r["rules_hash"],
                    "price_data_fingerprint": r["price_data_fingerprint"],
                    "status": r["status"],
                    "summary": json.loads(r["summary_json"]),
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                }
                for r in rows
            ]

    def upsert_trader_backtest_artifacts(
        self,
        *,
        run_id: str,
        trader_id: str,
        historical_trades_path: str,
        historical_pnl_path: str,
        weekly_signal_mask_path: str,
        weekly_returns_path: str,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trader_backtest_artifacts
                    (run_id, trader_id, historical_trades_path, historical_pnl_path,
                     weekly_signal_mask_path, weekly_returns_path, metadata_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    trader_id=excluded.trader_id,
                    historical_trades_path=excluded.historical_trades_path,
                    historical_pnl_path=excluded.historical_pnl_path,
                    weekly_signal_mask_path=excluded.weekly_signal_mask_path,
                    weekly_returns_path=excluded.weekly_returns_path,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at;
                """,
                (
                    run_id,
                    trader_id,
                    historical_trades_path,
                    historical_pnl_path,
                    weekly_signal_mask_path,
                    weekly_returns_path,
                    json.dumps(metadata or {}, ensure_ascii=True),
                    utc_now_iso(),
                ),
            )
            conn.commit()

    def get_trader_backtest_artifacts(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT run_id, trader_id, historical_trades_path, historical_pnl_path,
                       weekly_signal_mask_path, weekly_returns_path, metadata_json, updated_at
                FROM trader_backtest_artifacts
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "run_id": row["run_id"],
                "trader_id": row["trader_id"],
                "historical_trades_path": row["historical_trades_path"],
                "historical_pnl_path": row["historical_pnl_path"],
                "weekly_signal_mask_path": row["weekly_signal_mask_path"],
                "weekly_returns_path": row["weekly_returns_path"],
                "metadata": json.loads(row["metadata_json"]),
                "updated_at": row["updated_at"],
            }

    def replace_trader_weekly_signal_mask(
        self,
        *,
        run_id: str,
        trader_id: str,
        rows: List[Dict[str, Any]],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM trader_weekly_signal_mask WHERE run_id = ? AND trader_id = ?",
                (run_id, trader_id),
            )
            conn.executemany(
                """
                INSERT INTO trader_weekly_signal_mask
                    (run_id, trader_id, week_end, active, side, bars_in_market, pnl_week, mask_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        trader_id,
                        str(row.get("week_end") or ""),
                        int(bool(row.get("active"))),
                        str(row.get("side") or ""),
                        int(row.get("bars_in_market") or 0),
                        float(row.get("pnl_week") or 0.0),
                        str(row.get("mask_source") or "real_backtest"),
                    )
                    for row in rows
                ],
            )
            conn.commit()

    def list_latest_trader_weekly_signal_mask(self, trader_id: str) -> List[Dict[str, Any]]:
        latest = self.get_latest_trader_backtest_run(trader_id)
        if latest is None:
            return []
        run_id = str(latest["run_id"])
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id, trader_id, week_end, active, side, bars_in_market, pnl_week, mask_source
                FROM trader_weekly_signal_mask
                WHERE run_id = ? AND trader_id = ?
                ORDER BY week_end ASC
                """,
                (run_id, trader_id),
            ).fetchall()
            return [
                {
                    "run_id": r["run_id"],
                    "trader_id": r["trader_id"],
                    "week_end": r["week_end"],
                    "active": int(r["active"]),
                    "side": r["side"],
                    "bars_in_market": int(r["bars_in_market"]),
                    "pnl_week": float(r["pnl_week"]),
                    "mask_source": r["mask_source"],
                }
                for r in rows
            ]

    def replace_trader_weekly_returns(
        self,
        *,
        run_id: str,
        trader_id: str,
        rows: List[Dict[str, Any]],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM trader_weekly_returns WHERE run_id = ? AND trader_id = ?",
                (run_id, trader_id),
            )
            conn.executemany(
                """
                INSERT INTO trader_weekly_returns
                    (run_id, trader_id, week_end, weekly_return, equity_close, balance_close)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        trader_id,
                        str(row.get("week_end") or ""),
                        float(row.get("weekly_return") or 0.0),
                        float(row.get("equity_close") or 0.0),
                        float(row.get("balance_close") or 0.0),
                    )
                    for row in rows
                ],
            )
            conn.commit()

    def list_latest_trader_weekly_returns(self, trader_id: str) -> List[Dict[str, Any]]:
        latest = self.get_latest_trader_backtest_run(trader_id)
        if latest is None:
            return []
        run_id = str(latest["run_id"])
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id, trader_id, week_end, weekly_return, equity_close, balance_close
                FROM trader_weekly_returns
                WHERE run_id = ? AND trader_id = ?
                ORDER BY week_end ASC
                """,
                (run_id, trader_id),
            ).fetchall()
            return [
                {
                    "run_id": r["run_id"],
                    "trader_id": r["trader_id"],
                    "week_end": r["week_end"],
                    "weekly_return": float(r["weekly_return"]),
                    "equity_close": float(r["equity_close"]),
                    "balance_close": float(r["balance_close"]),
                }
                for r in rows
            ]

    def upsert_portfolio_universe_member(
        self,
        *,
        trader_id: str,
        asset: str,
        timeframe: str,
        promotion_date: str,
        lifecycle_state: str,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO portfolio_universe_registry
                    (trader_id, asset, timeframe, promotion_date, lifecycle_state, metadata_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trader_id) DO UPDATE SET
                    asset=excluded.asset,
                    timeframe=excluded.timeframe,
                    promotion_date=excluded.promotion_date,
                    lifecycle_state=excluded.lifecycle_state,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at;
                """,
                (
                    trader_id,
                    asset,
                    timeframe,
                    promotion_date,
                    lifecycle_state,
                    json.dumps(metadata or {}, ensure_ascii=True),
                    utc_now_iso(),
                ),
            )
            conn.commit()

    def list_portfolio_universe_members(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT trader_id, asset, timeframe, promotion_date, lifecycle_state, metadata_json, updated_at
                FROM portfolio_universe_registry
                ORDER BY promotion_date ASC, trader_id ASC
                """
            ).fetchall()
            return [
                {
                    "trader_id": r["trader_id"],
                    "asset": r["asset"],
                    "timeframe": r["timeframe"],
                    "promotion_date": r["promotion_date"],
                    "lifecycle_state": r["lifecycle_state"],
                    "metadata": json.loads(r["metadata_json"]),
                    "updated_at": r["updated_at"],
                }
                for r in rows
            ]

    def upsert_portfolio_training_run(
        self,
        *,
        run_id: str,
        run_type: str,
        model_version: str,
        status: str,
        started_at: str,
        completed_at: str = "",
        algorithm: str = "ppo",
        seed: int = 0,
        device: str = "cpu",
        hyperparameters: Dict[str, Any] | None = None,
        metrics: Dict[str, Any] | None = None,
        artifacts: Dict[str, Any] | None = None,
        notes: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO portfolio_training_runs
                    (run_id, run_type, model_version, status, started_at, completed_at, algorithm, seed, device,
                     hyperparameters_json, metrics_json, artifacts_json, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    run_type=excluded.run_type,
                    model_version=excluded.model_version,
                    status=excluded.status,
                    started_at=excluded.started_at,
                    completed_at=excluded.completed_at,
                    algorithm=excluded.algorithm,
                    seed=excluded.seed,
                    device=excluded.device,
                    hyperparameters_json=excluded.hyperparameters_json,
                    metrics_json=excluded.metrics_json,
                    artifacts_json=excluded.artifacts_json,
                    notes=excluded.notes;
                """,
                (
                    run_id,
                    run_type,
                    model_version,
                    status,
                    started_at,
                    completed_at,
                    algorithm,
                    int(seed),
                    device,
                    json.dumps(hyperparameters or {}, ensure_ascii=True),
                    json.dumps(metrics or {}, ensure_ascii=True),
                    json.dumps(artifacts or {}, ensure_ascii=True),
                    notes,
                ),
            )
            conn.commit()

    def list_portfolio_training_runs(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id, run_type, model_version, status, started_at, completed_at, algorithm, seed, device,
                       hyperparameters_json, metrics_json, artifacts_json, notes
                FROM portfolio_training_runs
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            return [
                {
                    "run_id": r["run_id"],
                    "run_type": r["run_type"],
                    "model_version": r["model_version"],
                    "status": r["status"],
                    "started_at": r["started_at"],
                    "completed_at": r["completed_at"],
                    "algorithm": r["algorithm"],
                    "seed": int(r["seed"]),
                    "device": r["device"],
                    "hyperparameters": json.loads(r["hyperparameters_json"]),
                    "metrics": json.loads(r["metrics_json"]),
                    "artifacts": json.loads(r["artifacts_json"]),
                    "notes": r["notes"],
                }
                for r in rows
            ]

    def upsert_portfolio_training_metric(
        self,
        *,
        run_id: str,
        step: int,
        split: str,
        metric_name: str,
        metric_value: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO portfolio_training_metrics
                    (run_id, step, split, metric_name, metric_value, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, step, split, metric_name) DO UPDATE SET
                    metric_value=excluded.metric_value,
                    recorded_at=excluded.recorded_at;
                """,
                (run_id, int(step), split, metric_name, float(metric_value), utc_now_iso()),
            )
            conn.commit()

    def list_portfolio_training_metrics(self, run_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id, step, split, metric_name, metric_value, recorded_at
                FROM portfolio_training_metrics
                WHERE run_id = ?
                ORDER BY step ASC, split ASC, metric_name ASC
                """,
                (run_id,),
            ).fetchall()
            return [
                {
                    "run_id": r["run_id"],
                    "step": int(r["step"]),
                    "split": r["split"],
                    "metric_name": r["metric_name"],
                    "metric_value": float(r["metric_value"]),
                    "recorded_at": r["recorded_at"],
                }
                for r in rows
            ]

    def upsert_portfolio_model_info(
        self,
        *,
        model_version: str,
        mode: str,
        checkpoint_path: str,
        universe_size: int,
        trained_at: str = "",
        fine_tuned_at: str = "",
        config: Dict[str, Any] | None = None,
        metrics: Dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO portfolio_model_registry
                    (model_version, mode, checkpoint_path, universe_size, trained_at, fine_tuned_at, config_json, metrics_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_version) DO UPDATE SET
                    mode=excluded.mode,
                    checkpoint_path=excluded.checkpoint_path,
                    universe_size=excluded.universe_size,
                    trained_at=excluded.trained_at,
                    fine_tuned_at=excluded.fine_tuned_at,
                    config_json=excluded.config_json,
                    metrics_json=excluded.metrics_json,
                    updated_at=excluded.updated_at;
                """,
                (
                    model_version,
                    mode,
                    checkpoint_path,
                    int(universe_size),
                    trained_at,
                    fine_tuned_at,
                    json.dumps(config or {}, ensure_ascii=True),
                    json.dumps(metrics or {}, ensure_ascii=True),
                    utc_now_iso(),
                ),
            )
            conn.commit()

    def get_latest_portfolio_model_info(self) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT model_version, mode, checkpoint_path, universe_size, trained_at, fine_tuned_at,
                       config_json, metrics_json, updated_at
                FROM portfolio_model_registry
                ORDER BY COALESCE(NULLIF(fine_tuned_at, ''), trained_at) DESC, updated_at DESC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            return {
                "model_version": row["model_version"],
                "mode": row["mode"],
                "checkpoint_path": row["checkpoint_path"],
                "universe_size": int(row["universe_size"]),
                "trained_at": row["trained_at"],
                "fine_tuned_at": row["fine_tuned_at"],
                "config": json.loads(row["config_json"]),
                "metrics": json.loads(row["metrics_json"]),
                "updated_at": row["updated_at"],
            }

    def upsert_portfolio_rebalance_snapshot(
        self,
        *,
        rebalance_id: str,
        rebalance_date: str,
        model_version: str,
        training_run_id: str = "",
        fine_tune_run_id: str = "",
        active_traders: List[str] | None = None,
        selected_traders: List[str] | None = None,
        target_weights: Dict[str, float] | None = None,
        target_cash_weight: float = 0.0,
        diagnostics: Dict[str, Any] | None = None,
        forward_metrics: Dict[str, Any] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO portfolio_rebalance_snapshots
                    (rebalance_id, rebalance_date, model_version, training_run_id, fine_tune_run_id,
                     active_traders_json, selected_traders_json, target_weights_json, target_cash_weight,
                     diagnostics_json, forward_metrics_json, metadata_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(rebalance_id) DO UPDATE SET
                    rebalance_date=excluded.rebalance_date,
                    model_version=excluded.model_version,
                    training_run_id=excluded.training_run_id,
                    fine_tune_run_id=excluded.fine_tune_run_id,
                    active_traders_json=excluded.active_traders_json,
                    selected_traders_json=excluded.selected_traders_json,
                    target_weights_json=excluded.target_weights_json,
                    target_cash_weight=excluded.target_cash_weight,
                    diagnostics_json=excluded.diagnostics_json,
                    forward_metrics_json=excluded.forward_metrics_json,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at;
                """,
                (
                    rebalance_id,
                    rebalance_date,
                    model_version,
                    training_run_id,
                    fine_tune_run_id,
                    json.dumps(active_traders or [], ensure_ascii=True),
                    json.dumps(selected_traders or [], ensure_ascii=True),
                    json.dumps(target_weights or {}, ensure_ascii=True),
                    float(target_cash_weight),
                    json.dumps(diagnostics or {}, ensure_ascii=True),
                    json.dumps(forward_metrics or {}, ensure_ascii=True),
                    json.dumps(metadata or {}, ensure_ascii=True),
                    utc_now_iso(),
                ),
            )
            conn.commit()

    def list_portfolio_rebalance_snapshots(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT rebalance_id, rebalance_date, model_version, training_run_id, fine_tune_run_id,
                       active_traders_json, selected_traders_json, target_weights_json, target_cash_weight,
                       diagnostics_json, forward_metrics_json, metadata_json, updated_at
                FROM portfolio_rebalance_snapshots
                ORDER BY rebalance_date DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            return [
                {
                    "rebalance_id": r["rebalance_id"],
                    "rebalance_date": r["rebalance_date"],
                    "model_version": r["model_version"],
                    "training_run_id": r["training_run_id"],
                    "fine_tune_run_id": r["fine_tune_run_id"],
                    "active_traders": json.loads(r["active_traders_json"]),
                    "selected_traders": json.loads(r["selected_traders_json"]),
                    "target_weights": json.loads(r["target_weights_json"]),
                    "target_cash_weight": float(r["target_cash_weight"]),
                    "diagnostics": json.loads(r["diagnostics_json"]),
                    "forward_metrics": json.loads(r["forward_metrics_json"]),
                    "metadata": json.loads(r["metadata_json"]),
                    "updated_at": r["updated_at"],
                }
                for r in rows
            ]

    def get_latest_portfolio_rebalance_snapshot(self) -> Optional[Dict[str, Any]]:
        rows = self.list_portfolio_rebalance_snapshots(limit=1)
        return rows[0] if rows else None

    def upsert_portfolio_forward_evaluation(
        self,
        *,
        evaluation_id: str,
        rebalance_id: str,
        benchmark_name: str,
        as_of: str,
        cumulative_return_1y: float,
        sharpe_1y: float,
        max_drawdown_1y: float,
        curve_points: List[Dict[str, Any]] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO portfolio_forward_evaluations
                    (evaluation_id, rebalance_id, benchmark_name, as_of, cumulative_return_1y,
                     sharpe_1y, max_drawdown_1y, curve_points_json, metadata_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evaluation_id) DO UPDATE SET
                    rebalance_id=excluded.rebalance_id,
                    benchmark_name=excluded.benchmark_name,
                    as_of=excluded.as_of,
                    cumulative_return_1y=excluded.cumulative_return_1y,
                    sharpe_1y=excluded.sharpe_1y,
                    max_drawdown_1y=excluded.max_drawdown_1y,
                    curve_points_json=excluded.curve_points_json,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at;
                """,
                (
                    evaluation_id,
                    rebalance_id,
                    benchmark_name,
                    as_of,
                    float(cumulative_return_1y),
                    float(sharpe_1y),
                    float(max_drawdown_1y),
                    json.dumps(curve_points or [], ensure_ascii=True),
                    json.dumps(metadata or {}, ensure_ascii=True),
                    utc_now_iso(),
                ),
            )
            conn.commit()

    def list_portfolio_forward_evaluations(
        self,
        *,
        rebalance_id: str | None = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            if rebalance_id:
                rows = conn.execute(
                    """
                    SELECT evaluation_id, rebalance_id, benchmark_name, as_of, cumulative_return_1y,
                           sharpe_1y, max_drawdown_1y, curve_points_json, metadata_json, updated_at
                    FROM portfolio_forward_evaluations
                    WHERE rebalance_id = ?
                    ORDER BY as_of DESC
                    LIMIT ?
                    """,
                    (rebalance_id, int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT evaluation_id, rebalance_id, benchmark_name, as_of, cumulative_return_1y,
                           sharpe_1y, max_drawdown_1y, curve_points_json, metadata_json, updated_at
                    FROM portfolio_forward_evaluations
                    ORDER BY as_of DESC
                    LIMIT ?
                    """,
                    (int(limit),),
                ).fetchall()
            return [
                {
                    "evaluation_id": r["evaluation_id"],
                    "rebalance_id": r["rebalance_id"],
                    "benchmark_name": r["benchmark_name"],
                    "as_of": r["as_of"],
                    "cumulative_return_1y": float(r["cumulative_return_1y"]),
                    "sharpe_1y": float(r["sharpe_1y"]),
                    "max_drawdown_1y": float(r["max_drawdown_1y"]),
                    "curve_points": json.loads(r["curve_points_json"]),
                    "metadata": json.loads(r["metadata_json"]),
                    "updated_at": r["updated_at"],
                }
                for r in rows
            ]

    def upsert_trader_design_profile(
        self,
        *,
        trader_id: str,
        asset: str,
        timeframe: str,
        promoted_at: str,
        profile: Dict[str, Any],
    ) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trader_design_profiles
                    (trader_id, asset, timeframe, promoted_at, profile_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trader_id) DO UPDATE SET
                    asset=excluded.asset,
                    timeframe=excluded.timeframe,
                    promoted_at=excluded.promoted_at,
                    profile_json=excluded.profile_json,
                    updated_at=excluded.updated_at;
                """,
                (trader_id, asset, timeframe, promoted_at, json.dumps(profile or {}, ensure_ascii=True), now, now),
            )
            conn.commit()

    def get_trader_design_profile(self, trader_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT trader_id, asset, timeframe, promoted_at, profile_json, created_at, updated_at
                FROM trader_design_profiles
                WHERE trader_id = ?
                """,
                (trader_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "trader_id": row["trader_id"],
                "asset": row["asset"],
                "timeframe": row["timeframe"],
                "promoted_at": row["promoted_at"],
                "profile": json.loads(row["profile_json"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }

    def list_trader_design_profiles(self, limit: int = 500) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT trader_id, asset, timeframe, promoted_at, profile_json, created_at, updated_at
                FROM trader_design_profiles
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            return [
                {
                    "trader_id": r["trader_id"],
                    "asset": r["asset"],
                    "timeframe": r["timeframe"],
                    "promoted_at": r["promoted_at"],
                    "profile": json.loads(r["profile_json"]),
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                }
                for r in rows
            ]

    def create_risk_evaluation_run(
        self,
        *,
        run_id: str,
        run_type: str,
        status: str = "running",
        notes: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO risk_evaluation_runs
                    (run_id, run_type, status, started_at, completed_at, evaluated_traders,
                     retraining_count, retrain_requests_count, notes, metadata_json)
                VALUES (?, ?, ?, ?, '', 0, 0, 0, ?, ?)
                """,
                (run_id, run_type, status, utc_now_iso(), notes, json.dumps(metadata or {}, ensure_ascii=True)),
            )
            conn.commit()

    def complete_risk_evaluation_run(
        self,
        *,
        run_id: str,
        status: str,
        evaluated_traders: int,
        retraining_count: int,
        retrain_requests_count: int,
        notes: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE risk_evaluation_runs
                SET status = ?,
                    completed_at = ?,
                    evaluated_traders = ?,
                    retraining_count = ?,
                    retrain_requests_count = ?,
                    notes = ?,
                    metadata_json = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    utc_now_iso(),
                    int(evaluated_traders),
                    int(retraining_count),
                    int(retrain_requests_count),
                    notes,
                    json.dumps(metadata or {}, ensure_ascii=True),
                    run_id,
                ),
            )
            conn.commit()

    def list_risk_evaluation_runs(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id, run_type, status, started_at, completed_at, evaluated_traders,
                       retraining_count, retrain_requests_count, notes, metadata_json
                FROM risk_evaluation_runs
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            return [
                {
                    "run_id": r["run_id"],
                    "run_type": r["run_type"],
                    "status": r["status"],
                    "started_at": r["started_at"],
                    "completed_at": r["completed_at"],
                    "evaluated_traders": int(r["evaluated_traders"]),
                    "retraining_count": int(r["retraining_count"]),
                    "retrain_requests_count": int(r["retrain_requests_count"]),
                    "notes": r["notes"],
                    "metadata": json.loads(r["metadata_json"]),
                }
                for r in rows
            ]

    def save_trader_forward_backtest_run(
        self,
        *,
        run_id: str,
        evaluation_run_id: str,
        trader_id: str,
        asset: str,
        timeframe: str,
        promoted_at: str,
        forward_start: str,
        forward_end: str,
        status: str,
        artifact_paths: Dict[str, Any] | None = None,
        metrics: Dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trader_forward_backtest_runs
                    (run_id, evaluation_run_id, trader_id, asset, timeframe, promoted_at, forward_start,
                     forward_end, status, artifact_paths_json, metrics_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    evaluation_run_id=excluded.evaluation_run_id,
                    trader_id=excluded.trader_id,
                    asset=excluded.asset,
                    timeframe=excluded.timeframe,
                    promoted_at=excluded.promoted_at,
                    forward_start=excluded.forward_start,
                    forward_end=excluded.forward_end,
                    status=excluded.status,
                    artifact_paths_json=excluded.artifact_paths_json,
                    metrics_json=excluded.metrics_json;
                """,
                (
                    run_id,
                    evaluation_run_id,
                    trader_id,
                    asset,
                    timeframe,
                    promoted_at,
                    forward_start,
                    forward_end,
                    status,
                    json.dumps(artifact_paths or {}, ensure_ascii=True),
                    json.dumps(metrics or {}, ensure_ascii=True),
                    utc_now_iso(),
                ),
            )
            conn.commit()

    def save_trader_forward_metrics(
        self,
        *,
        evaluation_run_id: str,
        trader_id: str,
        asset: str,
        timeframe: str,
        evaluation_date: str,
        metrics: Dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trader_forward_metrics
                    (evaluation_run_id, trader_id, asset, timeframe, evaluation_date, metrics_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation_run_id,
                    trader_id,
                    asset,
                    timeframe,
                    evaluation_date,
                    json.dumps(metrics or {}, ensure_ascii=True),
                    utc_now_iso(),
                ),
            )
            conn.commit()

    def list_trader_forward_metrics(
        self,
        *,
        trader_id: str | None = None,
        evaluation_run_id: str | None = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if trader_id:
            where.append("trader_id = ?")
            params.append(trader_id)
        if evaluation_run_id:
            where.append("evaluation_run_id = ?")
            params.append(evaluation_run_id)
        sql = """
            SELECT id, evaluation_run_id, trader_id, asset, timeframe, evaluation_date, metrics_json, created_at
            FROM trader_forward_metrics
        """
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY evaluation_date DESC, created_at DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
            return [
                {
                    "id": int(r["id"]),
                    "evaluation_run_id": r["evaluation_run_id"],
                    "trader_id": r["trader_id"],
                    "asset": r["asset"],
                    "timeframe": r["timeframe"],
                    "evaluation_date": r["evaluation_date"],
                    "metrics": json.loads(r["metrics_json"]),
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

    def list_trader_forward_backtest_runs(
        self,
        *,
        trader_id: str | None = None,
        evaluation_run_id: str | None = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if trader_id:
            where.append("trader_id = ?")
            params.append(trader_id)
        if evaluation_run_id:
            where.append("evaluation_run_id = ?")
            params.append(evaluation_run_id)
        sql = """
            SELECT run_id, evaluation_run_id, trader_id, asset, timeframe, promoted_at, forward_start, forward_end,
                   status, artifact_paths_json, metrics_json, created_at
            FROM trader_forward_backtest_runs
        """
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
            return [
                {
                    "run_id": r["run_id"],
                    "evaluation_run_id": r["evaluation_run_id"],
                    "trader_id": r["trader_id"],
                    "asset": r["asset"],
                    "timeframe": r["timeframe"],
                    "promoted_at": r["promoted_at"],
                    "forward_start": r["forward_start"],
                    "forward_end": r["forward_end"],
                    "status": r["status"],
                    "artifact_paths": json.loads(r["artifact_paths_json"]),
                    "metrics": json.loads(r["metrics_json"]),
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

    def save_risk_evaluation_detail(
        self,
        *,
        evaluation_run_id: str,
        trader_id: str,
        asset: str,
        timeframe: str,
        previous_state: str,
        new_state: str,
        action: str,
        health_score: float,
        reasons: List[str] | None = None,
        flags: Dict[str, Any] | None = None,
        retrain_request: Dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO risk_evaluation_details
                    (evaluation_run_id, trader_id, asset, timeframe, previous_state, new_state, action,
                     health_score, reasons_json, flags_json, retrain_request_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation_run_id,
                    trader_id,
                    asset,
                    timeframe,
                    previous_state,
                    new_state,
                    action,
                    float(health_score),
                    json.dumps(reasons or [], ensure_ascii=True),
                    json.dumps(flags or {}, ensure_ascii=True),
                    json.dumps(retrain_request or {}, ensure_ascii=True),
                    utc_now_iso(),
                ),
            )
            conn.commit()

    def list_risk_evaluation_details(
        self,
        *,
        evaluation_run_id: str | None = None,
        trader_id: str | None = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if evaluation_run_id:
            where.append("evaluation_run_id = ?")
            params.append(evaluation_run_id)
        if trader_id:
            where.append("trader_id = ?")
            params.append(trader_id)
        sql = """
            SELECT id, evaluation_run_id, trader_id, asset, timeframe, previous_state, new_state, action,
                   health_score, reasons_json, flags_json, retrain_request_json, created_at
            FROM risk_evaluation_details
        """
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
            return [
                {
                    "id": int(r["id"]),
                    "evaluation_run_id": r["evaluation_run_id"],
                    "trader_id": r["trader_id"],
                    "asset": r["asset"],
                    "timeframe": r["timeframe"],
                    "previous_state": r["previous_state"],
                    "new_state": r["new_state"],
                    "action": r["action"],
                    "health_score": float(r["health_score"]),
                    "reasons": json.loads(r["reasons_json"]),
                    "flags": json.loads(r["flags_json"]),
                    "retrain_request": json.loads(r["retrain_request_json"]),
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

    def create_retrain_request(
        self,
        *,
        request_id: str,
        trader_id: str,
        asset: str,
        timeframe: str,
        reason: str,
        priority: str,
        status: str = "pending",
        payload: Dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO retrain_requests
                    (request_id, trader_id, asset, timeframe, reason, priority, status,
                     created_at, consumed_at, completed_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, '', '', ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    trader_id=excluded.trader_id,
                    asset=excluded.asset,
                    timeframe=excluded.timeframe,
                    reason=excluded.reason,
                    priority=excluded.priority,
                    status=excluded.status,
                    payload_json=excluded.payload_json;
                """,
                (
                    request_id,
                    trader_id,
                    asset,
                    timeframe,
                    reason,
                    priority,
                    status,
                    utc_now_iso(),
                    json.dumps(payload or {}, ensure_ascii=True),
                ),
            )
            conn.commit()

    def list_pending_retrain_requests(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT request_id, trader_id, asset, timeframe, reason, priority, status,
                       created_at, consumed_at, completed_at, payload_json
                FROM retrain_requests
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            return [
                {
                    "request_id": r["request_id"],
                    "trader_id": r["trader_id"],
                    "asset": r["asset"],
                    "timeframe": r["timeframe"],
                    "reason": r["reason"],
                    "priority": r["priority"],
                    "status": r["status"],
                    "created_at": r["created_at"],
                    "consumed_at": r["consumed_at"],
                    "completed_at": r["completed_at"],
                    "payload": json.loads(r["payload_json"]),
                }
                for r in rows
            ]

    def list_retrain_requests(self, limit: int = 500) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT request_id, trader_id, asset, timeframe, reason, priority, status,
                       created_at, consumed_at, completed_at, payload_json
                FROM retrain_requests
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            return [
                {
                    "request_id": r["request_id"],
                    "trader_id": r["trader_id"],
                    "asset": r["asset"],
                    "timeframe": r["timeframe"],
                    "reason": r["reason"],
                    "priority": r["priority"],
                    "status": r["status"],
                    "created_at": r["created_at"],
                    "consumed_at": r["consumed_at"],
                    "completed_at": r["completed_at"],
                    "payload": json.loads(r["payload_json"]),
                }
                for r in rows
            ]

    def mark_retrain_request_running(self, request_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE retrain_requests
                SET status = 'running', consumed_at = ?
                WHERE request_id = ?
                """,
                (utc_now_iso(), request_id),
            )
            conn.commit()

    def mark_retrain_request_completed(self, request_id: str, payload: Dict[str, Any] | None = None) -> None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload_json FROM retrain_requests WHERE request_id = ?", (request_id,)).fetchone()
            current_payload = json.loads(row["payload_json"]) if row else {}
            merged_payload = {**current_payload, **(payload or {})}
            conn.execute(
                """
                UPDATE retrain_requests
                SET status = 'completed', completed_at = ?, payload_json = ?
                WHERE request_id = ?
                """,
                (utc_now_iso(), json.dumps(merged_payload, ensure_ascii=True), request_id),
            )
            conn.commit()

    def mark_retrain_request_failed(self, request_id: str, error: str) -> None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload_json FROM retrain_requests WHERE request_id = ?", (request_id,)).fetchone()
            current_payload = json.loads(row["payload_json"]) if row else {}
            current_payload["error"] = str(error)
            conn.execute(
                """
                UPDATE retrain_requests
                SET status = 'failed', completed_at = ?, payload_json = ?
                WHERE request_id = ?
                """,
                (utc_now_iso(), json.dumps(current_payload, ensure_ascii=True), request_id),
            )
            conn.commit()

    def save_risk_portfolio_check(
        self,
        *,
        check_id: str,
        rebalance_id: str,
        approved: bool,
        action: str,
        original_weights: Dict[str, float] | None = None,
        adjusted_weights: Dict[str, float] | None = None,
        blocked_traders: List[str] | None = None,
        clipped_traders: List[str] | None = None,
        reasons: List[str] | None = None,
        diagnostics: Dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO risk_portfolio_checks
                    (check_id, rebalance_id, created_at, approved, action, original_weights_json,
                     adjusted_weights_json, blocked_traders_json, clipped_traders_json,
                     reasons_json, diagnostics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(check_id) DO UPDATE SET
                    rebalance_id=excluded.rebalance_id,
                    created_at=excluded.created_at,
                    approved=excluded.approved,
                    action=excluded.action,
                    original_weights_json=excluded.original_weights_json,
                    adjusted_weights_json=excluded.adjusted_weights_json,
                    blocked_traders_json=excluded.blocked_traders_json,
                    clipped_traders_json=excluded.clipped_traders_json,
                    reasons_json=excluded.reasons_json,
                    diagnostics_json=excluded.diagnostics_json;
                """,
                (
                    check_id,
                    rebalance_id,
                    created_at or utc_now_iso(),
                    int(bool(approved)),
                    action,
                    json.dumps(original_weights or {}, ensure_ascii=True),
                    json.dumps(adjusted_weights or {}, ensure_ascii=True),
                    json.dumps(blocked_traders or [], ensure_ascii=True),
                    json.dumps(clipped_traders or [], ensure_ascii=True),
                    json.dumps(reasons or [], ensure_ascii=True),
                    json.dumps(diagnostics or {}, ensure_ascii=True),
                ),
            )
            conn.commit()

    def list_risk_portfolio_checks(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT check_id, rebalance_id, created_at, approved, action, original_weights_json,
                       adjusted_weights_json, blocked_traders_json, clipped_traders_json,
                       reasons_json, diagnostics_json
                FROM risk_portfolio_checks
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            return [
                {
                    "check_id": r["check_id"],
                    "rebalance_id": r["rebalance_id"],
                    "created_at": r["created_at"],
                    "approved": bool(r["approved"]),
                    "action": r["action"],
                    "original_weights": json.loads(r["original_weights_json"]),
                    "adjusted_weights": json.loads(r["adjusted_weights_json"]),
                    "blocked_traders": json.loads(r["blocked_traders_json"]),
                    "clipped_traders": json.loads(r["clipped_traders_json"]),
                    "reasons": json.loads(r["reasons_json"]),
                    "diagnostics": json.loads(r["diagnostics_json"]),
                }
                for r in rows
            ]

    def save_trader_signal_audit(
        self,
        *,
        timestamp: str,
        trader_id: str,
        asset: str,
        timeframe: str,
        signal_side: str,
        signal_active: bool,
        ppo_selected: bool,
        ppo_weight: float,
        risk_approved: bool,
        executed: bool,
        hypothetical_return: float = 0.0,
        executed_return: float = 0.0,
        reason_if_blocked: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trader_signal_audit
                    (timestamp, trader_id, asset, timeframe, signal_side, signal_active, ppo_selected,
                     ppo_weight, risk_approved, executed, hypothetical_return, executed_return,
                     reason_if_blocked, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    trader_id,
                    asset,
                    timeframe,
                    signal_side,
                    int(bool(signal_active)),
                    int(bool(ppo_selected)),
                    float(ppo_weight),
                    int(bool(risk_approved)),
                    int(bool(executed)),
                    float(hypothetical_return),
                    float(executed_return),
                    reason_if_blocked,
                    json.dumps(metadata or {}, ensure_ascii=True),
                ),
            )
            conn.commit()

    def list_trader_signal_audit(
        self,
        *,
        trader_id: str | None = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        sql = """
            SELECT id, timestamp, trader_id, asset, timeframe, signal_side, signal_active, ppo_selected,
                   ppo_weight, risk_approved, executed, hypothetical_return, executed_return,
                   reason_if_blocked, metadata_json
            FROM trader_signal_audit
        """
        params: list[Any] = []
        if trader_id:
            sql += " WHERE trader_id = ?"
            params.append(trader_id)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
            return [
                {
                    "id": int(r["id"]),
                    "timestamp": r["timestamp"],
                    "trader_id": r["trader_id"],
                    "asset": r["asset"],
                    "timeframe": r["timeframe"],
                    "signal_side": r["signal_side"],
                    "signal_active": bool(r["signal_active"]),
                    "ppo_selected": bool(r["ppo_selected"]),
                    "ppo_weight": float(r["ppo_weight"]),
                    "risk_approved": bool(r["risk_approved"]),
                    "executed": bool(r["executed"]),
                    "hypothetical_return": float(r["hypothetical_return"]),
                    "executed_return": float(r["executed_return"]),
                    "reason_if_blocked": r["reason_if_blocked"],
                    "metadata": json.loads(r["metadata_json"]),
                }
                for r in rows
            ]

