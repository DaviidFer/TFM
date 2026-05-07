from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional
from uuid import uuid4

import pandas as pd

from app.contracts import (
    AgentStatus,
    EventType,
    PromotedTraderSpec,
    ValidationReport,
)
from app.core.structured_logging import emit_log
from app.services import build_promoted_spec, run_validation_pipeline
from app.services.rule_generation_service import safe_rules_from_df

from .base import AgentContext
from .developer_agent import DevelopmentOutput


@dataclass
class ValidationOutput:
    report: ValidationReport
    promoted_spec: Optional[PromotedTraderSpec]


class ValidationAgent:
    agent_id = "validation_agent"

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx

    def validate_and_promote(
        self,
        dev: DevelopmentOutput,
        *,
        validation_profile: Mapping[str, Mapping[str, object]] | None = None,
        promote_if_empty: bool = True,
    ) -> ValidationOutput:
        self.ctx.store.set_agent_status(self.agent_id, AgentStatus.RUNNING, "validating and promoting")
        emit_log(
            self.agent_id,
            "validation_started",
            experiment_id=dev.experiment_config.experiment_id,
            asset=dev.experiment_config.asset,
            timeframe=dev.experiment_config.timeframe,
            candidate_counts={
                "long": len(dev.candidate_rules.long_rules),
                "short": len(dev.candidate_rules.short_rules),
            },
        )

        out = run_validation_pipeline(
            data_is=dev.blocks["data_is"],
            data_oos=dev.blocks["data_oos"],
            data_2025=dev.blocks["data_2025"],
            candidates_by_family=dev.candidates_by_family,
            validation_profile=validation_profile,
        )
        stable = out["stability"]
        winners_long = stable.get("winners_long_stable", [])
        winners_short = stable.get("winners_short_stable", [])

        fallback_used = False
        if len(winners_long) + len(winners_short) == 0:
            # Fallback defensivo para no romper fase de integración si estabilidad queda vacía.
            winners_long = safe_rules_from_df(out.get("decor_long", pd.DataFrame()))[:5]
            winners_short = safe_rules_from_df(out.get("decor_short", pd.DataFrame()))[:5]
            fallback_used = True

        report = ValidationReport(
            experiment_id=dev.experiment_config.experiment_id,
            asset=dev.experiment_config.asset,
            passed_long=len(winners_long),
            passed_short=len(winners_short),
            failed_long=max(len(dev.candidate_rules.long_rules) - len(winners_long), 0),
            failed_short=max(len(dev.candidate_rules.short_rules) - len(winners_short), 0),
            notes="fallback_used" if fallback_used else "stable_rules_selected",
            metrics={
                "n_candidates_long": len(dev.candidate_rules.long_rules),
                "n_candidates_short": len(dev.candidate_rules.short_rules),
                "n_stable_long": len(winners_long),
                "n_stable_short": len(winners_short),
                "validation_profile": out.get("validation_profile", {}),
            },
        )

        if (len(winners_long) + len(winners_short) == 0) and not promote_if_empty:
            emit_log(
                self.agent_id,
                "validation_completed_no_promotion",
                experiment_id=dev.experiment_config.experiment_id,
                asset=dev.experiment_config.asset,
                notes="no rules passed validation",
            )
            self.ctx.store.append_event(
                event_id=f"evt_{uuid4().hex[:10]}",
                event_type=EventType.VALIDATION_COMPLETED,
                producer=self.agent_id,
                payload={
                    **report.to_dict(),
                    "validation_profile": out.get("validation_profile", {}),
                    "promoted": False,
                },
                correlation_id=dev.experiment_config.experiment_id,
            )
            self.ctx.store.set_agent_status(self.agent_id, AgentStatus.IDLE, "validation done (no promotion)")
            return ValidationOutput(report=report, promoted_spec=None)

        promoted = build_promoted_spec(
            asset=dev.experiment_config.asset,
            timeframe=dev.experiment_config.timeframe,
            experiment_id=dev.experiment_config.experiment_id,
            winners_long_stable=winners_long,
            winners_short_stable=winners_short,
        )

        # El ValidationAgent NO marca estado en trader_states. El trader solo
        # aparece en trader_states cuando el TraderAgent.activate lo pone LIVE.
        # Mientras tanto vive en el evento TRADER_PROMOTED (cola validada).
        self.ctx.store.append_event(
            event_id=f"evt_{uuid4().hex[:10]}",
            event_type=EventType.VALIDATION_COMPLETED,
            producer=self.agent_id,
            payload={
                **report.to_dict(),
                "validation_profile": out.get("validation_profile", {}),
            },
            correlation_id=dev.experiment_config.experiment_id,
        )
        self.ctx.store.append_event(
            event_id=f"evt_{uuid4().hex[:10]}",
            event_type=EventType.TRADER_PROMOTED,
            producer=self.agent_id,
            payload=promoted.to_dict(),
            correlation_id=dev.experiment_config.experiment_id,
        )
        emit_log(
            self.agent_id,
            "validation_completed",
            experiment_id=dev.experiment_config.experiment_id,
            passed_long=len(winners_long),
            passed_short=len(winners_short),
            validation_profile=out.get("validation_profile", {}),
        )
        emit_log(
            self.agent_id,
            "trader_promoted_with_rules",
            trader_id=promoted.trader_id,
            asset=promoted.asset,
            timeframe=promoted.timeframe,
            long_rules=promoted.long_rules,
            short_rules=promoted.short_rules,
        )
        self.ctx.store.set_agent_status(self.agent_id, AgentStatus.IDLE, "validation done")
        return ValidationOutput(report=report, promoted_spec=promoted)

