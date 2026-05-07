"""
Orquestador para smoke checks (`app/phase5_check.py`).

El flujo operativo real usa `DevelopmentOperationalSupervisor`
(`app/runtime/development_operational_supervisor.py`). Este módulo conserva el
ciclo DataProcess → Developer → Validation → Trader ante eventos de retraining.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, Mapping

from app.agents import DeveloperAgent, TraderAgent, ValidationAgent
from app.contracts import EventType
from app.core.structured_logging import emit_log
from app.services import DataProcess


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RuntimeOrchestrator:
    """
    Orquestador pragmático de Fase 5.
    - consume eventos de retraining
    - dispara ciclo DataProcess->Developer->Validation->Trader
    """

    def __init__(
        self,
        *,
        data_process: DataProcess,
        developer_agent: DeveloperAgent,
        validation_agent: ValidationAgent,
        trader_agent: TraderAgent,
    ) -> None:
        self.data_process = data_process
        self.developer_agent = developer_agent
        self.validation_agent = validation_agent
        self.trader_agent = trader_agent
        self._processed_event_ids: set[str] = set()

    def handle_retrain_request(
        self,
        *,
        asset: str,
        timeframe: str,
        asset_csv_path: str,
        families: tuple[str, ...],
        family_params: Mapping[str, Mapping[str, object]] | None = None,
    ) -> Dict[str, str]:
        emit_log(
            "runtime_orchestrator",
            "retrain_handle_started",
            asset=asset,
            timeframe=timeframe,
            asset_csv_path=asset_csv_path,
            families=list(families),
            family_params=dict(family_params or {}),
        )
        dataset = self.data_process.prepare_dataset(
            asset=asset,
            timeframe=timeframe,
            asset_csv_path=asset_csv_path,
        )
        dev = self.developer_agent.develop(
            dataset=dataset,
            families=families,
            family_params=family_params,
        )
        val = self.validation_agent.validate_and_promote(dev)
        metrics = self.trader_agent.activate(val.promoted_spec)
        out = {
            "dataset_id": dataset.dataset_id,
            "experiment_id": dev.experiment_config.experiment_id,
            "trader_id": val.promoted_spec.trader_id,
            "as_of": metrics.as_of,
        }
        emit_log("runtime_orchestrator", "retrain_handle_completed", output=out)
        return out

    def process_pending_retrain_events(
        self,
        *,
        asset_csv_by_asset: Mapping[str, str],
        families: tuple[str, ...],
        family_params: Mapping[str, Mapping[str, object]] | None = None,
    ) -> list[Dict[str, str]]:
        store = self.data_process.ctx.store
        events = store.list_events(limit=500)
        results: list[Dict[str, str]] = []
        emit_log(
            "runtime_orchestrator",
            "retrain_scan_started",
            events_loaded=len(events),
            already_processed=len(self._processed_event_ids),
        )

        # oldest first for deterministic behavior
        for e in reversed(events):
            if e["event_id"] in self._processed_event_ids:
                continue
            if e["event_type"] != EventType.RETRAIN_REQUESTED.value:
                continue
            payload = e["payload"]
            asset = str(payload.get("asset"))
            timeframe = str(payload.get("timeframe", "D1"))
            csv_path = asset_csv_by_asset.get(asset)
            if not csv_path:
                emit_log(
                    "runtime_orchestrator",
                    "retrain_skipped_missing_asset_csv",
                    trigger_event_id=e["event_id"],
                    asset=asset,
                )
                continue
            out = self.handle_retrain_request(
                asset=asset,
                timeframe=timeframe,
                asset_csv_path=csv_path,
                families=families,
                family_params=family_params,
            )
            out["trigger_event_id"] = e["event_id"]
            results.append(out)
            self._processed_event_ids.add(e["event_id"])

            store.append_event(
                event_id=f"evt_{uuid.uuid4().hex[:10]}",
                event_type=EventType.RETRAIN_PROCESSED,
                producer="runtime_orchestrator",
                payload={
                    "message": "retrain request processed",
                    "trigger_event_id": e["event_id"],
                    "new_trader_id": out["trader_id"],
                    "processed_at": utc_now_iso(),
                },
                correlation_id=e.get("correlation_id"),
            )
        emit_log(
            "runtime_orchestrator",
            "retrain_scan_completed",
            processed_count=len(results),
            processed_trigger_event_ids=[r["trigger_event_id"] for r in results],
        )
        return results
