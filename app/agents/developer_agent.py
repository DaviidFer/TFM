from __future__ import annotations

from dataclasses import dataclass
import random
from time import perf_counter
from typing import Dict, List, Mapping, Tuple
from uuid import uuid4

import pandas as pd

from app.contracts import AgentStatus, CandidateRules, EventType, ExperimentConfig
from app.core.structured_logging import emit_log
from app.services.data_service import PreparedDataset
from app.services.feature_service import build_features
from app.services.rule_generation_service import (
    FAMILIES_WITH_RULE_TARGET,
    generate_candidate_rules,
    safe_rules_from_df,
)
from app.services.split_service import split_is_oos_holdout
from app.services.target_service import apply_target_to_blocks

from .base import AgentContext


def _date_range(df: pd.DataFrame) -> dict[str, str | None]:
    if df is None or df.empty:
        return {"start": None, "end": None}
    idx = df.index
    return {"start": idx.min().isoformat(), "end": idx.max().isoformat()}


def _normalize_split_cfg(split_config: Mapping[str, object] | None) -> Dict[str, object]:
    """Aplica defaults deterministas al split.

    `is_recent` deja de ser aleatorio dentro del agente: el supervisor lo fija
    explícitamente, y los phase checks que no lo pasan obtienen un valor
    estable (`True` = IS reciente / OOS antiguo).
    """
    cfg = dict(split_config or {})
    is_recent = cfg.get("is_recent", None)
    return {
        "is_pct": float(cfg.get("is_pct", 0.5)),
        "oos_pct": float(cfg.get("oos_pct", 0.5)),
        "holdout_year": int(cfg.get("holdout_year", 2025)),
        "lookback_years": int(cfg.get("lookback_years", 10)),
        "is_recent": True if is_recent is None else bool(is_recent),
    }


def _prepare_blocks(
    dataset: PreparedDataset,
    split_cfg: Mapping[str, object],
) -> Dict[str, pd.DataFrame]:
    """`raw OHLC -> features -> split IS/OOS/holdout -> target` en un solo paso."""
    raw = dataset.ohlc.copy()
    features = build_features(data_ohlc=raw)
    blocks = split_is_oos_holdout(
        features,
        is_pct=float(split_cfg["is_pct"]),
        oos_pct=float(split_cfg["oos_pct"]),
        holdout_year=int(split_cfg["holdout_year"]),
        lookback_years=int(split_cfg["lookback_years"]),
        is_recent=bool(split_cfg["is_recent"]),
    )
    return apply_target_to_blocks(blocks)


def _collect_rules(
    *,
    blocks: Mapping[str, pd.DataFrame],
    families: tuple[str, ...],
    base_params: Mapping[str, Mapping[str, object]],
    target_long: int,
    target_short: int,
    max_iterations: int,
) -> Tuple[
    List[str],
    List[str],
    Dict[str, Mapping[str, pd.DataFrame]],
    List[Dict[str, object]],
    Dict[str, Dict[str, int]],
    str,
]:
    """Bucle iterativo de generación con deduplicación.

    Devuelve `(selected_long, selected_short, last_candidates_by_family,
    iteration_summaries, accumulated_family_counts)`. Sin overflow cruzado:
    si tras `max_iterations` no se alcanza `target_long+target_short`, se
    reporta el total alcanzado tal cual (sin rellenar con reglas del lado
    contrario) para no romper el invariante long/short.
    """
    selected_long: List[str] = []
    selected_short: List[str] = []
    seen_long: set[str] = set()
    seen_short: set[str] = set()
    last_candidates_by_family: Dict[str, Mapping[str, pd.DataFrame]] = {}
    iteration_summaries: List[Dict[str, object]] = []
    accumulated: Dict[str, Dict[str, int]] = {}
    stop_reason = "max_iterations"

    target_with_rule_target = set(FAMILIES_WITH_RULE_TARGET)

    for iteration in range(1, max_iterations + 1):
        remaining_long = max(0, target_long - len(selected_long))
        remaining_short = max(0, target_short - len(selected_short))
        if remaining_long <= 0 and remaining_short <= 0:
            stop_reason = "targets_reached"
            break

        batch_target = max(12, min(48, remaining_long + remaining_short))
        iter_params: Dict[str, Dict[str, object]] = {
            k: dict(v or {}) for k, v in base_params.items()
        }
        for fam in target_with_rule_target:
            if fam in iter_params:
                iter_params[fam]["target_n_rules"] = int(batch_target)

        iteration_started = perf_counter()
        candidates_by_family = generate_candidate_rules(
            data_is=blocks["data_is"],
            families=families,
            family_params=iter_params,
        )
        last_candidates_by_family = candidates_by_family

        family_rule_counts: Dict[str, Dict[str, int]] = {}
        for fam, grp in candidates_by_family.items():
            n_long = int(len(grp.get("long", pd.DataFrame())))
            n_short = int(len(grp.get("short", pd.DataFrame())))
            family_rule_counts[fam] = {"long": n_long, "short": n_short}
            slot = accumulated.setdefault(fam, {"long": 0, "short": 0})
            slot["long"] += n_long
            slot["short"] += n_short

        new_unique_long = 0
        new_unique_short = 0
        iteration_summaries.append(
            {
                "iteration": iteration,
                "batch_target": int(batch_target),
                "family_rule_counts": family_rule_counts,
            }
        )

        for grp in candidates_by_family.values():
            for rule in safe_rules_from_df(grp.get("long", pd.DataFrame())):
                if rule in seen_long:
                    continue
                seen_long.add(rule)
                new_unique_long += 1
                if len(selected_long) < target_long:
                    selected_long.append(rule)
            for rule in safe_rules_from_df(grp.get("short", pd.DataFrame())):
                if rule in seen_short:
                    continue
                seen_short.add(rule)
                new_unique_short += 1
                if len(selected_short) < target_short:
                    selected_short.append(rule)

        iteration_summaries[-1]["new_unique_long"] = int(new_unique_long)
        iteration_summaries[-1]["new_unique_short"] = int(new_unique_short)
        iteration_summaries[-1]["elapsed_ms"] = int((perf_counter() - iteration_started) * 1000)
        if new_unique_long == 0 and new_unique_short == 0:
            iteration_summaries[-1]["stop_reason"] = "no_new_rules"
            stop_reason = "no_new_rules"
            break

    return (
        selected_long,
        selected_short,
        last_candidates_by_family,
        iteration_summaries,
        accumulated,
        stop_reason,
    )


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
        dataset: PreparedDataset,
        families: tuple[str, ...],
        family_params: Mapping[str, Mapping[str, object]] | None = None,
        split_config: Mapping[str, object] | None = None,
    ) -> DevelopmentOutput:
        self.ctx.store.set_agent_status(self.agent_id, AgentStatus.RUNNING, "building candidates")
        split_cfg = _normalize_split_cfg(split_config)
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

        develop_started = perf_counter()
        split_started = perf_counter()
        blocks = _prepare_blocks(dataset, split_cfg)
        split_elapsed_ms = int((perf_counter() - split_started) * 1000)
        split_detail = {
            "is_pct": split_cfg["is_pct"],
            "oos_pct": split_cfg["oos_pct"],
            "holdout_year": split_cfg["holdout_year"],
            "lookback_years": split_cfg["lookback_years"],
            "is_orientation": "is_recent_oos_oldest" if bool(split_cfg["is_recent"]) else "is_oldest_oos_recent",
        }
        block_rows = {k: int(len(v)) for k, v in blocks.items()}
        block_date_ranges = {k: _date_range(v) for k, v in blocks.items()}
        emit_log(
            self.agent_id,
            "split_and_target_ready",
            split_policy="is_oos_holdout_2025",
            split_detail=split_detail,
            block_rows=block_rows,
            block_date_ranges=block_date_ranges,
        )
        self.ctx.store.append_event(
            event_id=f"evt_{uuid4().hex[:10]}",
            event_type=EventType.SPLIT_AND_TARGET_READY,
            producer=self.agent_id,
            payload={
                "asset": dataset.asset,
                "timeframe": dataset.timeframe,
                "split_policy": "is_oos_holdout_2025",
                "split_detail": split_detail,
                "block_rows": block_rows,
                "block_date_ranges": block_date_ranges,
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
                "is_recent": bool(split_cfg["is_recent"]),
                "family_params": dict(family_params or {}),
            },
        )

        target_total_rules = 100
        long_ratio = random.uniform(0.3, 0.7)
        target_long = int(round(target_total_rules * long_ratio))
        target_short = int(target_total_rules - target_long)
        max_iterations = 12

        (
            selected_long,
            selected_short,
            last_candidates_by_family,
            iteration_summaries,
            accumulated_family_counts,
            generation_stop_reason,
        ) = _collect_rules(
            blocks=blocks,
            families=families,
            base_params=dict(family_params or {}),
            target_long=target_long,
            target_short=target_short,
            max_iterations=max_iterations,
        )
        generation_elapsed_ms = int((perf_counter() - develop_started) * 1000) - split_elapsed_ms
        timing_ms = {
            "split_and_target": max(split_elapsed_ms, 0),
            "rule_generation": max(generation_elapsed_ms, 0),
            "total": int((perf_counter() - develop_started) * 1000),
        }

        emit_log(
            self.agent_id,
            "rule_generation_selected_models",
            selected_families=list(last_candidates_by_family.keys()),
            family_rule_counts=accumulated_family_counts,
            family_params=dict(family_params or {}),
            iterations=iteration_summaries,
            stop_reason=generation_stop_reason,
            timing_ms=timing_ms,
        )

        candidates = CandidateRules(
            experiment_id=exp_id,
            asset=dataset.asset,
            long_rules=selected_long,
            short_rules=selected_short,
            generation_summary={
                "families": list(last_candidates_by_family.keys()),
                "n_long": len(selected_long),
                "n_short": len(selected_short),
                "n_total": len(selected_long) + len(selected_short),
                "target_total": target_total_rules,
                "target_long": target_long,
                "target_short": target_short,
                "long_ratio_target": round(long_ratio, 4),
                "iterations": len(iteration_summaries),
                "iteration_summaries": iteration_summaries,
                "family_rule_counts_accumulated": accumulated_family_counts,
                "generation_stop_reason": generation_stop_reason,
                "timing_ms": timing_ms,
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
            candidates_by_family=last_candidates_by_family,
        )
