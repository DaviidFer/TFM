from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

import pandas as pd

from app.contracts import (
    AgentStatus,
    EventType,
    PromotedTraderSpec,
    TraderDesignProfile,
    TraderHealthConfig,
    TraderHealthSnapshot,
    TraderLifecycleState,
    TraderReviewAction,
)
from app.core.structured_logging import emit_log
from app.services.trader_health import (
    ForwardBacktestService,
    build_trader_design_profile,
    evaluate_trader_health,
)

from .base import AgentContext


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class HumanResourcesProcess:
    """
    `HumanResourcesProcess` supervisa periodicamente a los traders promovidos.

    Para cada trader, reconstruye su perfil de diseno, recalcula su comportamiento
    forward post-promocion y compara ambos. Si el trader sigue alineado con su
    comportamiento esperado, continua operativo (`KEEP`). Si deja de estarlo, se
    marca para reentrenamiento (`RETRAINING`) y se emite un `RetrainRequest` para
    generar un sustituto.

    Este proceso NO interviene en la cartera: no revisa decisiones del
    `PortfolioManagerProcess`, no ajusta pesos, no controla margen ni broker, y
    no actua como gate pre-trade.
    """

    agent_id = "human_resources_process"

    def __init__(
        self,
        ctx: AgentContext,
        *,
        config: Optional[TraderHealthConfig] = None,
        data_source: Any | None = None,
    ) -> None:
        self.ctx = ctx
        self.config = config or TraderHealthConfig()
        self.data_source = data_source or getattr(ctx.execution_router, "market_data", None)
        self.forward_service = ForwardBacktestService(store=self.ctx.store, artifacts_root=self.ctx.artifacts_root)

    # ------------------------------------------------------------------ helpers

    def _load_promoted_specs(self) -> Dict[str, PromotedTraderSpec]:
        specs: Dict[str, PromotedTraderSpec] = {}
        try:
            events = self.ctx.store.list_events_by_type(EventType.TRADER_PROMOTED.value, limit=50000)
        except Exception:
            return specs
        for event in events:
            payload = dict(event.get("payload") or {})
            trader_id = str(payload.get("trader_id") or "")
            asset = str(payload.get("asset") or "").upper()
            if not trader_id or not asset:
                continue
            try:
                specs[trader_id] = PromotedTraderSpec(
                    trader_id=trader_id,
                    asset=asset,
                    timeframe=str(payload.get("timeframe") or "D1"),
                    long_rules=list(payload.get("long_rules") or []),
                    short_rules=list(payload.get("short_rules") or []),
                    origin_experiment_id=str(payload.get("origin_experiment_id") or event.get("correlation_id") or f"hr_{trader_id}"),
                    promoted_at=str(payload.get("promoted_at") or event.get("occurred_at") or _utc_now_iso()),
                    metadata=dict(payload.get("metadata") or {}),
                )
            except Exception:
                continue
        return specs

    def _build_design_profile(self, spec: PromotedTraderSpec) -> TraderDesignProfile:
        row = self.ctx.store.get_trader_design_profile(str(spec.trader_id))
        if row is not None:
            payload = dict(row.get("profile") or {})
            return TraderDesignProfile(**payload)
        latest_run = self.ctx.store.get_latest_trader_backtest_run(str(spec.trader_id)) or {}
        artifacts = self.ctx.store.get_trader_backtest_artifacts(str(latest_run.get("run_id") or "")) or {}
        profile = build_trader_design_profile(
            promoted_spec=spec,
            validation_report=None,
            backtest_summary=latest_run,
            backtest_artifacts=artifacts,
        )
        self.ctx.store.upsert_trader_design_profile(
            trader_id=str(profile.trader_id),
            asset=str(profile.asset),
            timeframe=str(profile.timeframe),
            promoted_at=str(profile.promoted_at),
            profile=profile.to_dict(),
        )
        return profile

    def _persist_snapshot(self, snapshot: TraderHealthSnapshot) -> None:
        self.ctx.store.save_trader_review_detail(
            evaluation_run_id=snapshot.evaluation_run_id,
            trader_id=snapshot.trader_id,
            asset=snapshot.asset,
            timeframe=snapshot.timeframe,
            previous_state=snapshot.previous_state,
            new_state=snapshot.new_state,
            action=snapshot.action,
            health_score=snapshot.health_score,
            reasons=list(snapshot.reasons),
            flags=dict(snapshot.flags),
            retrain_request=snapshot.retrain_request.to_dict() if snapshot.retrain_request is not None else {},
        )
        self.ctx.store.upsert_trader_state(
            trader_id=snapshot.trader_id,
            asset=snapshot.asset,
            timeframe=snapshot.timeframe,
            state=TraderLifecycleState(snapshot.new_state),
            notes="; ".join(snapshot.reasons),
        )
        self.ctx.store.append_event(
            event_id=f"evt_{uuid4().hex[:10]}",
            event_type=EventType.TRADER_HEALTH_EVALUATED,
            producer=self.agent_id,
            payload=snapshot.to_dict(),
            correlation_id=snapshot.evaluation_run_id,
        )
        if snapshot.retrain_request is not None:
            priority = "high" if snapshot.action == TraderReviewAction.RETRAINING.value else "normal"
            self.ctx.store.create_retrain_request(
                request_id=snapshot.retrain_request.request_id,
                trader_id=snapshot.retrain_request.trader_id,
                asset=snapshot.retrain_request.asset,
                timeframe=snapshot.retrain_request.timeframe,
                reason=snapshot.retrain_request.reason,
                priority=priority,
                status="pending",
                payload=snapshot.retrain_request.to_dict(),
            )
            self.ctx.store.append_event(
                event_id=f"evt_{uuid4().hex[:10]}",
                event_type=EventType.RETRAIN_REQUESTED,
                producer=self.agent_id,
                payload=snapshot.retrain_request.to_dict(),
                correlation_id=snapshot.retrain_request.request_id,
            )

    # ------------------------------------------------------------------ API

    def evaluate_single_trader(
        self,
        trader_id: str,
        *,
        evaluation_date: Optional[datetime] = None,
        run_type: str = "manual",
        force_backtest: bool = False,
        evaluation_run_id: Optional[str] = None,
    ) -> Optional[TraderHealthSnapshot]:
        """
        Evalua la salud de un unico trader: reconstruye perfil de diseno, corre
        backtest forward post-promocion, calcula health_score y decide KEEP /
        RETRAINING. Persiste el resultado y, si procede, emite RetrainRequest.
        """
        specs = self._load_promoted_specs()
        spec = specs.get(str(trader_id))
        if spec is None:
            return None
        state_row = self.ctx.store.get_trader_state(str(trader_id))
        current_state = state_row.state.value if state_row is not None else TraderLifecycleState.LIVE.value
        design_profile = self._build_design_profile(spec)
        eval_date = evaluation_date or datetime.now(timezone.utc)
        eval_iso = pd.Timestamp(eval_date).isoformat()
        run_id = str(evaluation_run_id or f"hr_{uuid4().hex[:10]}")
        forward_metrics = self.forward_service.run_forward_backtest_for_trader(
            trader_id=str(trader_id),
            promoted_spec=spec,
            design_profile=design_profile,
            evaluation_date=eval_iso,
            data_source=self.data_source,
            artifacts_root=self.ctx.artifacts_root,
            evaluation_run_id=run_id,
        )
        snapshot = evaluate_trader_health(
            design_profile=design_profile,
            forward_metrics=forward_metrics,
            current_state=current_state,
            config=self.config,
        )
        self._persist_snapshot(snapshot)
        emit_log(
            self.agent_id,
            "trader_health_evaluated",
            console=False,
            trader_id=str(trader_id),
            evaluation_run_id=run_id,
            action=snapshot.action,
            health_score=snapshot.health_score,
            reasons=snapshot.reasons,
            run_type=run_type,
            force_backtest=bool(force_backtest),
        )
        return snapshot

    def evaluate_trader_universe(
        self,
        evaluation_date: Optional[datetime] = None,
        run_type: str = "scheduled_monthly",
        force_backtest: bool = False,
    ) -> list[TraderHealthSnapshot]:
        """
        Lanza una revision de salud sobre TODOS los traders promovidos.
        Persiste un `trader_review_run` con su resumen agregado.
        """
        run_id = f"hr_{uuid4().hex[:10]}"
        self.ctx.store.set_agent_status(self.agent_id, AgentStatus.RUNNING, f"trader review running ({run_type})")
        self.ctx.store.create_trader_review_run(
            run_id=run_id,
            run_type=run_type,
            status="running",
            metadata={"force_backtest": bool(force_backtest)},
        )
        snapshots: list[TraderHealthSnapshot] = []
        retraining_count = 0
        retrain_requests = 0
        try:
            for trader_id in sorted(self._load_promoted_specs().keys()):
                snapshot = self.evaluate_single_trader(
                    trader_id,
                    evaluation_date=evaluation_date,
                    run_type=run_type,
                    force_backtest=force_backtest,
                    evaluation_run_id=run_id,
                )
                if snapshot is None:
                    continue
                snapshots.append(snapshot)
                if snapshot.action == TraderReviewAction.RETRAINING.value:
                    retraining_count += 1
                if snapshot.retrain_request is not None:
                    retrain_requests += 1
            self.ctx.store.complete_trader_review_run(
                run_id=run_id,
                status="completed",
                evaluated_traders=len(snapshots),
                retraining_count=retraining_count,
                retrain_requests_count=retrain_requests,
                metadata={"force_backtest": bool(force_backtest)},
            )
            self.ctx.store.set_agent_status(self.agent_id, AgentStatus.IDLE, "trader review completed")
            return snapshots
        except Exception as exc:
            self.ctx.store.complete_trader_review_run(
                run_id=run_id,
                status="failed",
                evaluated_traders=len(snapshots),
                retraining_count=retraining_count,
                retrain_requests_count=retrain_requests,
                notes=str(exc),
                metadata={"force_backtest": bool(force_backtest)},
            )
            self.ctx.store.set_agent_status(self.agent_id, AgentStatus.FAILED, str(exc))
            raise

    def should_run_monthly_evaluation(self, *, as_of: Optional[datetime] = None, force: bool = False) -> bool:
        """
        Politica simple: lanzar revision los primeros 3 dias del mes, o si han
        pasado mas de 30 dias desde la ultima evaluacion. `force=True` la lanza
        siempre.
        """
        if force:
            return True
        ref = pd.Timestamp(as_of or datetime.now(timezone.utc))
        latest = self.ctx.store.list_trader_review_runs(limit=1)
        if not latest:
            return True
        latest_ts = pd.Timestamp(str(latest[0].get("started_at") or ref.isoformat()))
        return int(ref.day) <= 3 or (ref - latest_ts).days >= 30

    def force_evaluation(self, *, evaluation_date: Optional[datetime] = None, force_backtest: bool = True) -> list[TraderHealthSnapshot]:
        """Atajo para lanzar una revision inmediata desde la UI o un job manual."""
        return self.evaluate_trader_universe(
            evaluation_date=evaluation_date or datetime.now(timezone.utc),
            run_type="forced",
            force_backtest=force_backtest,
        )
