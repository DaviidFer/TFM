from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping
from uuid import uuid4

import pandas as pd

from app.contracts import AgentStatus, CandidateRules, DatasetContract, EventType, ExperimentConfig
from app.core.structured_logging import emit_log
from app.services import apply_target_to_blocks, build_features, generate_candidate_rules, split_is_oos_holdout

from .base import AgentContext


def _safe_rules_from_df(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []
    col = "regla" if "regla" in df.columns else ("rule" if "rule" in df.columns else None)
    if col is None:
        return []
    return df[col].dropna().astype(str).tolist()


def _date_range(df: pd.DataFrame) -> dict[str, str | None]:
    if df is None or df.empty:
        return {"start": None, "end": None}
    idx = df.index
    return {"start": idx.min().isoformat(), "end": idx.max().isoformat()}


@dataclass
class DevelopmentOutput:
    experiment_config: ExperimentConfig
    candidate_rules: CandidateRules
    blocks: Dict[str, pd.DataFrame]
    candidates_by_family: Dict[str, Mapping[str, pd.DataFrame]]


class DeveloperAgent:
    agent_id = "developer_agent"

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx

    def develop(
        self,
        *,
        dataset: DatasetContract,
        families: tuple[str, ...],
        family_params: Mapping[str, Mapping[str, object]] | None = None,
        split_config: Mapping[str, object] | None = None,
    ) -> DevelopmentOutput:
        self.ctx.store.set_agent_status(self.agent_id, AgentStatus.RUNNING, "building candidates")
        split_cfg = {
            "is_pct": float((split_config or {}).get("is_pct", 0.5)),
            "oos_pct": float((split_config or {}).get("oos_pct", 0.5)),
            "holdout_year": int((split_config or {}).get("holdout_year", 2025)),
            "lookback_years": int((split_config or {}).get("lookback_years", 10)),
        }
        emit_log(
            self.agent_id,
            "development_started",
            dataset_id=dataset.dataset_id,
            asset=dataset.asset,
            timeframe=dataset.timeframe,
            families=list(families),
            family_params=dict(family_params or {}),
            split_config=split_cfg,
        )
        self.ctx.store.append_event(
            event_id=f"evt_{uuid4().hex[:10]}",
            event_type=EventType.DEVELOPMENT_STARTED,
            producer=self.agent_id,
            payload={
                "dataset_id": dataset.dataset_id,
                "asset": dataset.asset,
                "timeframe": dataset.timeframe,
                "families": list(families),
                "family_params": dict(family_params or {}),
                "split_config": split_cfg,
            },
            correlation_id=dataset.dataset_id,
        )

        raw = pd.read_csv(dataset.source_path)
        raw = raw.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close"})
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        raw = raw.dropna(subset=["date"]).sort_values("date").set_index("date")
        for c in ["open", "high", "low", "close"]:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        raw = raw.dropna(subset=["open", "high", "low", "close"])
        emit_log(
            self.agent_id,
            "raw_data_prepared",
            source_path=dataset.source_path,
            rows=int(len(raw)),
            start_date=raw.index.min().isoformat() if len(raw) else None,
            end_date=raw.index.max().isoformat() if len(raw) else None,
        )

        features = build_features(data_ohlc=raw)
        blocks = split_is_oos_holdout(
            features,
            is_pct=split_cfg["is_pct"],
            oos_pct=split_cfg["oos_pct"],
            holdout_year=split_cfg["holdout_year"],
            lookback_years=split_cfg["lookback_years"],
        )
        blocks = apply_target_to_blocks(blocks)
        emit_log(
            self.agent_id,
            "split_and_target_ready",
            split_policy="is_oos_holdout_2025",
            split_detail={
                "is_pct": split_cfg["is_pct"],
                "oos_pct": split_cfg["oos_pct"],
                "holdout_year": split_cfg["holdout_year"],
                "lookback_years": split_cfg["lookback_years"],
                "is_orientation": "inverted_split_is_recent_oos_is_oldest",
            },
            block_rows={k: int(len(v)) for k, v in blocks.items()},
            block_date_ranges={k: _date_range(v) for k, v in blocks.items()},
        )
        self.ctx.store.append_event(
            event_id=f"evt_{uuid4().hex[:10]}",
            event_type=EventType.SPLIT_AND_TARGET_READY,
            producer=self.agent_id,
            payload={
                "asset": dataset.asset,
                "timeframe": dataset.timeframe,
                "split_policy": "is_oos_holdout_2025",
                "split_detail": {
                    "is_pct": split_cfg["is_pct"],
                    "oos_pct": split_cfg["oos_pct"],
                    "holdout_year": split_cfg["holdout_year"],
                    "lookback_years": split_cfg["lookback_years"],
                    "is_orientation": "inverted_split_is_recent_oos_is_oldest",
                },
                "block_rows": {k: int(len(v)) for k, v in blocks.items()},
                "block_date_ranges": {k: _date_range(v) for k, v in blocks.items()},
            },
            correlation_id=dataset.dataset_id,
        )

        exp_id = f"exp_{dataset.asset.lower()}_{uuid4().hex[:8]}"
        experiment = ExperimentConfig(
            experiment_id=exp_id,
            asset=dataset.asset,
            timeframe=dataset.timeframe,
            split_policy="is_oos_holdout_2025",
            model_families=list(families),
            parameters={
                "is_pct": split_cfg["is_pct"],
                "oos_pct": split_cfg["oos_pct"],
                "holdout_year": split_cfg["holdout_year"],
                "lookback_years": split_cfg["lookback_years"],
                "family_params": dict(family_params or {}),
            },
        )

        candidates_by_family = generate_candidate_rules(
            data_is=blocks["data_is"],
            families=families,
            family_params=family_params,
        )
        family_rule_counts = {
            fam: {
                "long": int(len(grp.get("long", pd.DataFrame()))),
                "short": int(len(grp.get("short", pd.DataFrame()))),
            }
            for fam, grp in candidates_by_family.items()
        }
        emit_log(
            self.agent_id,
            "rule_generation_selected_models",
            selected_families=list(candidates_by_family.keys()),
            family_rule_counts=family_rule_counts,
            family_params=dict(family_params or {}),
        )

        all_long: list[str] = []
        all_short: list[str] = []
        for grp in candidates_by_family.values():
            all_long.extend(_safe_rules_from_df(grp.get("long", pd.DataFrame())))
            all_short.extend(_safe_rules_from_df(grp.get("short", pd.DataFrame())))

        candidates = CandidateRules(
            experiment_id=exp_id,
            asset=dataset.asset,
            long_rules=list(dict.fromkeys(all_long)),
            short_rules=list(dict.fromkeys(all_short)),
            generation_summary={
                "families": list(candidates_by_family.keys()),
                "n_long": len(all_long),
                "n_short": len(all_short),
            },
        )

        self.ctx.store.append_event(
            event_id=f"evt_{uuid4().hex[:10]}",
            event_type=EventType.CANDIDATE_RULES_READY,
            producer=self.agent_id,
            payload={
                "experiment": experiment.to_dict(),
                "candidate_summary": candidates.generation_summary,
            },
            correlation_id=exp_id,
        )
        emit_log(
            self.agent_id,
            "candidate_rules_ready",
            experiment_id=exp_id,
            asset=dataset.asset,
            summary=candidates.generation_summary,
        )
        self.ctx.store.set_agent_status(self.agent_id, AgentStatus.IDLE, "candidates ready")

        return DevelopmentOutput(
            experiment_config=experiment,
            candidate_rules=candidates,
            blocks=blocks,
            candidates_by_family=candidates_by_family,
        )

