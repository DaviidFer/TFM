from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional
from uuid import uuid4

import pandas as pd

from app.contracts import (
    AgentKind,
    AgentStatus,
    DesignRiskProfile,
    EventType,
    PortfolioDecision,
    RetrainRequest,
    RiskAction,
    RiskDecision,
    RiskLimitsConfig,
    RiskAdjustedPortfolioDecision,
    TraderForwardMetrics,
    TraderHealthSnapshot,
    TraderLifecycleState,
)
from app.core.structured_logging import emit_log
from app.services.risk import ForwardBacktestService, build_design_risk_profile, evaluate_trader_health

from .base import AgentContext


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RiskThresholds:
    max_drawdown: float = 0.25
    min_sharpe: float = -0.75
    min_trades: int = 5


class RiskAgent:
    agent_id = "risk_agent"

    def __init__(
        self,
        ctx: AgentContext,
        thresholds: Optional[RiskThresholds] = None,
        *,
        limits: Optional[RiskLimitsConfig] = None,
        data_source: Any | None = None,
    ) -> None:
        self.ctx = ctx
        self.thresholds = thresholds or RiskThresholds()
        self.limits = limits or RiskLimitsConfig(
            max_weight_per_trader=0.15,
            max_weight_per_asset=0.30,
            max_total_exposure=1.0,
            min_cash_buffer=0.10,
        )
        if thresholds is not None:
            self.limits = RiskLimitsConfig(
                **{
                    **self.limits.to_dict(),
                    "max_drawdown_multiplier_retire": max(float(self.thresholds.max_drawdown), 0.01),
                    "min_sharpe_ratio_suspend": max(float(self.thresholds.min_sharpe), -10.0),
                    "min_forward_trades_for_hard_decision": int(self.thresholds.min_trades),
                }
            )
        self.data_source = data_source or getattr(ctx.execution_router, "market_data", None)
        self.forward_service = ForwardBacktestService(store=self.ctx.store, artifacts_root=self.ctx.artifacts_root)

    def _load_promoted_specs(self) -> Dict[str, Any]:
        specs: Dict[str, Any] = {}
        try:
            events = self.ctx.store.list_events_by_type(EventType.TRADER_PROMOTED.value, limit=50000)
        except Exception:
            return specs
        from app.contracts import PromotedTraderSpec

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
                    origin_experiment_id=str(payload.get("origin_experiment_id") or event.get("correlation_id") or f"risk_{trader_id}"),
                    promoted_at=str(payload.get("promoted_at") or event.get("occurred_at") or utc_now_iso()),
                    metadata=dict(payload.get("metadata") or {}),
                )
            except Exception:
                continue
        return specs

    def _build_design_profile(self, spec: Any) -> DesignRiskProfile:
        row = self.ctx.store.get_trader_design_profile(str(spec.trader_id))
        if row is not None:
            payload = dict(row.get("profile") or {})
            return DesignRiskProfile(**payload)
        latest_run = self.ctx.store.get_latest_trader_backtest_run(str(spec.trader_id)) or {}
        artifacts = self.ctx.store.get_trader_backtest_artifacts(str(latest_run.get("run_id") or "")) or {}
        profile = build_design_risk_profile(
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

    def _append_risk_event(self, snapshot: TraderHealthSnapshot) -> None:
        self.ctx.store.append_event(
            event_id=f"evt_{uuid4().hex[:10]}",
            event_type=EventType.RISK_DECISION,
            producer=self.agent_id,
            payload=snapshot.to_dict(),
            correlation_id=snapshot.evaluation_run_id,
        )

    def _persist_snapshot(self, snapshot: TraderHealthSnapshot) -> None:
        self.ctx.store.save_risk_evaluation_detail(
            evaluation_run_id=snapshot.evaluation_run_id,
            trader_id=snapshot.trader_id,
            asset=snapshot.asset,
            timeframe=snapshot.timeframe,
            previous_state=snapshot.previous_state,
            new_state=snapshot.new_state,
            action=snapshot.action,
            health_score=snapshot.health_score,
            reasons=list(snapshot.reasons),
            flags=dict(snapshot.risk_flags),
            retrain_request=snapshot.retrain_request.to_dict() if snapshot.retrain_request is not None else {},
        )
        self.ctx.store.upsert_trader_state(
            trader_id=snapshot.trader_id,
            asset=snapshot.asset,
            timeframe=snapshot.timeframe,
            state=TraderLifecycleState(snapshot.new_state),
            notes="; ".join(snapshot.reasons),
        )
        self._append_risk_event(snapshot)
        if snapshot.retrain_request is not None:
            priority = "high" if snapshot.action == RiskAction.RETIRE.value else "normal"
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

    def _build_portfolio_state_map(self) -> Dict[str, Dict[str, Any]]:
        state_map: Dict[str, Dict[str, Any]] = {}
        for row in self.ctx.store.list_trader_states():
            state_map[str(row.trader_id)] = {
                "state": row.state.value,
                "asset": row.asset,
                "timeframe": row.timeframe,
                "updated_at": row.updated_at,
                "notes": row.notes,
            }
        return state_map

    def evaluate_single_trader(
        self,
        trader_id: str,
        *,
        evaluation_date: Optional[datetime] = None,
        run_type: str = "manual",
        force_backtest: bool = False,
        evaluation_run_id: Optional[str] = None,
    ) -> Optional[TraderHealthSnapshot]:
        specs = self._load_promoted_specs()
        spec = specs.get(str(trader_id))
        if spec is None:
            return None
        state_row = self.ctx.store.get_trader_state(str(trader_id))
        current_state = state_row.state.value if state_row is not None else TraderLifecycleState.PROMOTED.value
        design_profile = self._build_design_profile(spec)
        eval_date = evaluation_date or datetime.now(timezone.utc)
        eval_iso = pd.Timestamp(eval_date).isoformat()
        run_id = str(evaluation_run_id or f"risk_{uuid4().hex[:10]}")
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
            limits=self.limits,
        )
        self._persist_snapshot(snapshot)
        emit_log(
            self.agent_id,
            "risk_trader_evaluated",
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
        run_id = f"risk_{uuid4().hex[:10]}"
        self.ctx.store.set_agent_status(self.agent_id, AgentStatus.RUNNING, f"risk evaluating universe ({run_type})")
        self.ctx.store.create_risk_evaluation_run(run_id=run_id, run_type=run_type, status="running", metadata={"force_backtest": bool(force_backtest)})
        snapshots: list[TraderHealthSnapshot] = []
        degraded_count = 0
        suspended_count = 0
        retired_count = 0
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
                if snapshot.action == RiskAction.DEGRADED.value:
                    degraded_count += 1
                elif snapshot.action == RiskAction.SUSPEND.value:
                    suspended_count += 1
                elif snapshot.action == RiskAction.RETIRE.value:
                    retired_count += 1
                if snapshot.retrain_request is not None:
                    retrain_requests += 1
            self.ctx.store.complete_risk_evaluation_run(
                run_id=run_id,
                status="completed",
                evaluated_traders=len(snapshots),
                degraded_count=degraded_count,
                suspended_count=suspended_count,
                retired_count=retired_count,
                retrain_requests_count=retrain_requests,
                metadata={"force_backtest": bool(force_backtest)},
            )
            self.ctx.store.set_agent_status(self.agent_id, AgentStatus.IDLE, "risk evaluation completed")
            return snapshots
        except Exception as exc:
            self.ctx.store.complete_risk_evaluation_run(
                run_id=run_id,
                status="failed",
                evaluated_traders=len(snapshots),
                degraded_count=degraded_count,
                suspended_count=suspended_count,
                retired_count=retired_count,
                retrain_requests_count=retrain_requests,
                notes=str(exc),
                metadata={"force_backtest": bool(force_backtest)},
            )
            self.ctx.store.set_agent_status(self.agent_id, AgentStatus.FAILED, str(exc))
            raise

    def should_run_monthly_evaluation(self, *, as_of: Optional[datetime] = None, force: bool = False) -> bool:
        if force:
            return True
        ref = pd.Timestamp(as_of or datetime.now(timezone.utc))
        latest = self.ctx.store.list_risk_evaluation_runs(limit=1)
        if not latest:
            return True
        latest_ts = pd.Timestamp(str(latest[0].get("started_at") or ref.isoformat()))
        return int(ref.day) <= 3 or (ref - latest_ts).days >= 30

    def force_risk_evaluation(self, *, evaluation_date: Optional[datetime] = None, force_backtest: bool = True) -> list[TraderHealthSnapshot]:
        return self.evaluate_trader_universe(
            evaluation_date=evaluation_date or datetime.now(timezone.utc),
            run_type="forced",
            force_backtest=force_backtest,
        )

    def review_portfolio_decision(
        self,
        portfolio_decision: PortfolioDecision,
        account_info: Optional[dict],
        open_positions: Optional[list],
        limits: Optional[RiskLimitsConfig] = None,
    ) -> RiskAdjustedPortfolioDecision:
        del open_positions  # reservado para extensiones de riesgo adicionales
        limits = limits or self.limits
        state_map = self._build_portfolio_state_map()
        original_weights = {str(k): float(v) for k, v in dict(portfolio_decision.weights or {}).items()}
        adjusted = dict(original_weights)
        blocked: list[str] = []
        clipped: list[str] = []
        reasons: list[str] = []
        diagnostics: Dict[str, Any] = {"state_map": state_map}
        approved = True
        scaled_down = False

        if account_info:
            balance = float(account_info.get("balance") or 0.0)
            equity = float(account_info.get("equity") or balance or 0.0)
            diagnostics["account_info"] = {"balance": balance, "equity": equity, "margin_level": account_info.get("margin_level")}
            if limits.emergency_drawdown_stop is not None and balance > 0:
                account_dd = max(0.0, (balance - equity) / balance)
                diagnostics["account_drawdown"] = account_dd
                if account_dd >= float(limits.emergency_drawdown_stop):
                    adjusted = {k: 0.0 for k in adjusted}
                    approved = False
                    reasons.append("Emergency stop por drawdown de cuenta.")
                    result = RiskAdjustedPortfolioDecision(
                        rebalance_id=str(portfolio_decision.decision_id),
                        evaluation_id=f"rpc_{uuid4().hex[:10]}",
                        original_decision=portfolio_decision,
                        approved=False,
                        action=RiskAction.EMERGENCY_STOP.value,
                        adjusted_weights=adjusted,
                        original_weights=original_weights,
                        forced_cash_weight=1.0,
                        blocked_traders=list(original_weights.keys()),
                        clipped_traders=[],
                        scaled_down=False,
                        reasons=reasons,
                        diagnostics=diagnostics,
                    )
                    self.ctx.store.save_risk_portfolio_check(
                        check_id=result.evaluation_id,
                        rebalance_id=result.rebalance_id,
                        approved=result.approved,
                        action=result.action,
                        original_weights=result.original_weights,
                        adjusted_weights=result.adjusted_weights,
                        blocked_traders=result.blocked_traders,
                        clipped_traders=result.clipped_traders,
                        reasons=result.reasons,
                        diagnostics=result.diagnostics,
                        created_at=result.created_at,
                    )
                    return result

        for trader_id in list(adjusted.keys()):
            state = str((state_map.get(trader_id) or {}).get("state") or TraderLifecycleState.LIVE.value)
            if state in {TraderLifecycleState.SUSPENDED.value, TraderLifecycleState.RETIRED.value, TraderLifecycleState.RETRAINING.value}:
                if adjusted.get(trader_id, 0.0) > 0:
                    blocked.append(trader_id)
                    reasons.append(f"{trader_id} bloqueado por estado {state}.")
                adjusted[trader_id] = float(limits.suspended_weight if state == TraderLifecycleState.SUSPENDED.value else limits.retired_weight)
            elif state == TraderLifecycleState.DEGRADED.value:
                old = float(adjusted.get(trader_id, 0.0))
                new = old * float(limits.degraded_weight_multiplier)
                if new < old:
                    reasons.append(f"{trader_id} reducido por estado degraded.")
                    clipped.append(trader_id)
                adjusted[trader_id] = new

        for trader_id, weight in list(adjusted.items()):
            capped = min(float(weight), float(limits.max_weight_per_trader))
            if capped < float(weight):
                clipped.append(trader_id)
                reasons.append(f"{trader_id} recortado por límite max_weight_per_trader.")
            adjusted[trader_id] = max(0.0, capped)

        by_asset: Dict[str, list[str]] = {}
        for trader_id in adjusted:
            asset = str((state_map.get(trader_id) or {}).get("asset") or trader_id)
            by_asset.setdefault(asset, []).append(trader_id)
        for asset, trader_ids in by_asset.items():
            total = sum(float(adjusted.get(tid, 0.0)) for tid in trader_ids)
            if total > float(limits.max_weight_per_asset) > 0:
                scale = float(limits.max_weight_per_asset) / total
                for tid in trader_ids:
                    adjusted[tid] = float(adjusted[tid]) * scale
                    if tid not in clipped:
                        clipped.append(tid)
                reasons.append(f"Exposición por activo recortada en {asset}.")

        total_exposure = sum(float(v) for v in adjusted.values())
        max_exposure = min(float(limits.max_total_exposure), max(0.0, 1.0 - float(limits.min_cash_buffer)))
        if total_exposure > max_exposure > 0:
            scale = max_exposure / total_exposure
            adjusted = {k: float(v) * scale for k, v in adjusted.items()}
            total_exposure = sum(adjusted.values())
            scaled_down = True
            reasons.append("Exposición total reducida para respetar buffer de caja.")

        margin_level = None if not account_info else account_info.get("margin_level")
        if limits.min_broker_margin_level is not None and margin_level is not None:
            try:
                if float(margin_level) < float(limits.min_broker_margin_level):
                    adjusted = {k: 0.0 for k in adjusted}
                    total_exposure = 0.0
                    approved = False
                    reasons.append("Cartera rechazada por margen insuficiente.")
            except Exception:
                pass

        action = RiskAction.APPROVE.value
        if not approved and total_exposure == 0.0:
            action = RiskAction.REJECT_PORTFOLIO.value
        elif total_exposure == 0.0 and original_weights:
            action = RiskAction.FORCE_CASH.value
        elif scaled_down:
            action = RiskAction.SCALE_DOWN.value
        elif clipped:
            action = RiskAction.APPROVE_WITH_CLIPPING.value

        forced_cash_weight = max(float(portfolio_decision.target_cash_weight or 0.0), max(0.0, 1.0 - sum(adjusted.values())))
        result = RiskAdjustedPortfolioDecision(
            rebalance_id=str(portfolio_decision.decision_id),
            evaluation_id=f"rpc_{uuid4().hex[:10]}",
            original_decision=portfolio_decision,
            approved=approved,
            action=action,
            adjusted_weights={k: float(v) for k, v in adjusted.items() if float(v) > 0.0},
            original_weights=original_weights,
            forced_cash_weight=float(forced_cash_weight),
            blocked_traders=sorted(set(blocked)),
            clipped_traders=sorted(set(clipped)),
            scaled_down=bool(scaled_down),
            reasons=reasons or ["Cartera aprobada sin ajustes relevantes."],
            diagnostics={**diagnostics, "total_exposure": total_exposure, "limits": limits.to_dict()},
        )
        self.ctx.store.save_risk_portfolio_check(
            check_id=result.evaluation_id,
            rebalance_id=result.rebalance_id,
            approved=result.approved,
            action=result.action,
            original_weights=result.original_weights,
            adjusted_weights=result.adjusted_weights,
            blocked_traders=result.blocked_traders,
            clipped_traders=result.clipped_traders,
            reasons=result.reasons,
            diagnostics=result.diagnostics,
            created_at=result.created_at,
        )
        return result

    def assess_trader(self, trader_id: str, *, asset: str, timeframe: str) -> Optional[RiskDecision]:
        self.ctx.store.set_agent_status(self.agent_id, AgentStatus.RUNNING, f"assessing {trader_id}")
        emit_log(
            self.agent_id,
            "risk_assessment_started",
            trader_id=trader_id,
            asset=asset,
            timeframe=timeframe,
            thresholds={
                "max_drawdown": self.thresholds.max_drawdown,
                "min_sharpe": self.thresholds.min_sharpe,
                "min_trades": self.thresholds.min_trades,
            },
        )
        m = self.ctx.store.get_trader_metrics(trader_id)
        if m is None:
            emit_log(self.agent_id, "risk_assessment_skipped_no_metrics", trader_id=trader_id)
            self.ctx.store.set_agent_status(self.agent_id, AgentStatus.IDLE, "no metrics")
            return None

        dd = float(m["drawdown_rolling"])
        sharpe = float(m["sharpe_rolling"])
        trades = int(m["trade_count"])

        action = "keep"
        reason = "within thresholds"
        if dd >= self.thresholds.max_drawdown:
            action = "retire"
            reason = f"drawdown breach ({dd:.4f} >= {self.thresholds.max_drawdown:.4f})"
        elif trades >= self.thresholds.min_trades and sharpe <= self.thresholds.min_sharpe:
            action = "retire"
            reason = f"sharpe degradation ({sharpe:.4f} <= {self.thresholds.min_sharpe:.4f})"
        elif dd >= (0.8 * self.thresholds.max_drawdown):
            action = "suspend"
            reason = f"drawdown warning ({dd:.4f})"

        decision = RiskDecision(
            decision_id=f"rk_{uuid4().hex[:10]}",
            trader_id=trader_id,
            as_of=utc_now_iso(),
            action=action,
            reason=reason,
            triggered_metrics={
                "drawdown_rolling": dd,
                "sharpe_rolling": sharpe,
                "trade_count": trades,
            },
        )
        self.ctx.store.append_event(
            event_id=f"evt_{uuid4().hex[:10]}",
            event_type=EventType.RISK_DECISION,
            producer=self.agent_id,
            payload=decision.to_dict(),
            correlation_id=decision.decision_id,
        )
        emit_log(
            self.agent_id,
            "risk_decision",
            decision=decision.to_dict(),
        )

        if action == "retire":
            self.ctx.store.upsert_trader_state(
                trader_id=trader_id,
                asset=asset,
                timeframe=timeframe,
                state=TraderLifecycleState.RETIRED,
                notes=reason,
            )
            retrain = RetrainRequest(
                request_id=f"rr_{uuid4().hex[:10]}",
                trader_id=trader_id,
                asset=asset,
                timeframe=timeframe,
                reason=reason,
                requested_by=AgentKind.RISK,
                context={"risk_decision_id": decision.decision_id},
            )
            self.ctx.store.append_event(
                event_id=f"evt_{uuid4().hex[:10]}",
                event_type=EventType.RETRAIN_REQUESTED,
                producer=self.agent_id,
                payload=retrain.to_dict(),
                correlation_id=retrain.request_id,
            )
            self.ctx.store.upsert_trader_state(
                trader_id=trader_id,
                asset=asset,
                timeframe=timeframe,
                state=TraderLifecycleState.RETRAINING,
                notes="retrain requested by risk",
            )
        elif action == "suspend":
            self.ctx.store.upsert_trader_state(
                trader_id=trader_id,
                asset=asset,
                timeframe=timeframe,
                state=TraderLifecycleState.SUSPENDED,
                notes=reason,
            )

        self.ctx.store.set_agent_status(self.agent_id, AgentStatus.IDLE, action)
        return decision

    def get_broker_account_info(self) -> dict:
        if self.ctx.execution_router is None:
            return {"configured": False}
        return self.ctx.execution_router.get_account_info(actor=self.agent_id)

    def get_broker_positions(self) -> list[dict]:
        if self.ctx.execution_router is None:
            return []
        return self.ctx.execution_router.get_open_positions(actor=self.agent_id)

