from __future__ import annotations

from uuid import uuid4

from app.contracts import AgentStatus, DatasetContract, EventType
from app.core.structured_logging import emit_log
from app.services import load_asset_ohlc

from .base import AgentContext


class DataAgent:
    agent_id = "data_agent"

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx

    def prepare_dataset(self, *, asset: str, timeframe: str, asset_csv_path: str) -> DatasetContract:
        self.ctx.store.set_agent_status(self.agent_id, AgentStatus.RUNNING, "loading OHLC")
        emit_log(
            self.agent_id,
            "dataset_load_started",
            asset=asset,
            timeframe=timeframe,
            source_path=asset_csv_path,
        )
        ohlc = load_asset_ohlc(asset_csv_path=asset_csv_path)

        dataset = DatasetContract(
            dataset_id=f"ds_{asset.lower()}_{timeframe.lower()}_{uuid4().hex[:8]}",
            asset=asset,
            timeframe=timeframe,
            source_path=asset_csv_path,
            rows=len(ohlc),
            start_date=str(ohlc.index.min().date()),
            end_date=str(ohlc.index.max().date()),
            quality_score=1.0,
            metadata={"prepared_by": self.agent_id},
        )

        self.ctx.store.append_event(
            event_id=f"evt_{uuid4().hex[:10]}",
            event_type=EventType.DATASET_READY,
            producer=self.agent_id,
            payload=dataset.to_dict(),
            correlation_id=dataset.dataset_id,
        )
        emit_log(
            self.agent_id,
            "dataset_ready",
            dataset_id=dataset.dataset_id,
            asset=dataset.asset,
            timeframe=dataset.timeframe,
            rows=dataset.rows,
            start_date=dataset.start_date,
            end_date=dataset.end_date,
            quality_score=dataset.quality_score,
        )
        self.ctx.store.set_agent_status(self.agent_id, AgentStatus.IDLE, "dataset ready")
        return dataset

