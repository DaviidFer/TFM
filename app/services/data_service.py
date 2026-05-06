from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict
from uuid import uuid4

import pandas as pd

from app.contracts import AgentStatus, EventType
from app.core.structured_logging import emit_log

if TYPE_CHECKING:
    from app.agents.base import AgentContext


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class PreparedDataset:
    dataset_id: str
    asset: str
    timeframe: str
    source_path: str
    rows: int
    start_date: str
    end_date: str
    quality_score: float
    created_at: str = field(default_factory=_utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)
    ohlc: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False, compare=False)

    def event_payload(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "asset": self.asset,
            "timeframe": self.timeframe,
            "source_path": self.source_path,
            "rows": self.rows,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "quality_score": self.quality_score,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


class DataProcess:
    """
    Proceso de datos mínimo para la fase inicial:
    CSV sucio -> OHLC limpio -> dataset preparado para Developer.
    """

    agent_id = "data_process"

    def __init__(self, ctx: "AgentContext") -> None:
        self.ctx = ctx

    def prepare_dataset(self, *, asset: str, timeframe: str, asset_csv_path: str) -> PreparedDataset:
        self.ctx.store.set_agent_status(self.agent_id, AgentStatus.RUNNING, "loading CSV and cleaning OHLC")
        emit_log(
            self.agent_id,
            "dataset_load_started",
            asset=asset,
            timeframe=timeframe,
            source_path=asset_csv_path,
        )
        ohlc = load_asset_ohlc(asset_csv_path=asset_csv_path)

        dataset = PreparedDataset(
            dataset_id=f"ds_{asset.lower()}_{timeframe.lower()}_{uuid4().hex[:8]}",
            asset=asset,
            timeframe=timeframe,
            source_path=str(asset_csv_path),
            rows=len(ohlc),
            start_date=str(ohlc.index.min().date()),
            end_date=str(ohlc.index.max().date()),
            quality_score=1.0,
            metadata={"prepared_by": self.agent_id},
            ohlc=ohlc,
        )

        self.ctx.store.append_event(
            event_id=f"evt_{uuid4().hex[:10]}",
            event_type=EventType.DATASET_READY,
            producer=self.agent_id,
            payload=dataset.event_payload(),
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


def load_asset_ohlc(asset_csv_path: str | Path) -> pd.DataFrame:
    """
    Carga un CSV estilo Yahoo y devuelve OHLC normalizado:
    index: DatetimeIndex
    columns: open, high, low, close
    """
    p = Path(asset_csv_path)
    if not p.exists():
        raise FileNotFoundError(f"No existe CSV de activo: {p}")

    df = pd.read_csv(p)
    rename_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
    }
    df = df.rename(columns=rename_map)
    needed = ["date", "open", "high", "low", "close"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV sin columnas requeridas {missing}: {p}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").set_index("date")

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    out = df.dropna(subset=["open", "high", "low", "close"]).copy()
    if out.empty:
        raise ValueError(f"OHLC vacío tras limpieza: {p}")
    return out

