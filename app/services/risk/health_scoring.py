from __future__ import annotations

from typing import Any, Dict
from uuid import uuid4

from app.contracts import (
    AgentKind,
    DesignRiskProfile,
    RetrainRequest,
    RiskAction,
    RiskLimitsConfig,
    TraderForwardMetrics,
    TraderHealthSnapshot,
    TraderLifecycleState,
)
from app.contracts.models import utc_now_iso


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _apply_ratio_penalty(
    score: float,
    *,
    current: float | None,
    baseline: float | None,
    degraded_floor: float,
    suspend_floor: float,
    label: str,
    reasons: list[str],
    flags: Dict[str, Any],
    degraded_penalty: float = 22.0,
    suspend_penalty: float = 35.0,
) -> float:
    if baseline is None or current is None:
        return score
    if baseline == 0:
        return score
    ratio = current / baseline
    flags[f"{label}_ratio"] = ratio
    if ratio < suspend_floor:
        reasons.append(f"{label} forward muy por debajo del perfil de diseño.")
        return score - suspend_penalty
    if ratio < degraded_floor:
        reasons.append(f"{label} forward deteriorado frente al diseño.")
        return score - degraded_penalty
    return score


def evaluate_trader_health(
    design_profile: DesignRiskProfile,
    forward_metrics: TraderForwardMetrics,
    current_state: str,
    limits: RiskLimitsConfig,
) -> TraderHealthSnapshot:
    score = 100.0
    reasons: list[str] = []
    flags: Dict[str, Any] = {"insufficient_evidence": bool(forward_metrics.insufficient_evidence)}

    shadow_trades = int(forward_metrics.shadow_trades or 0)
    if shadow_trades < int(limits.min_forward_trades_for_hard_decision):
        score -= 8.0
        reasons.append("Evidencia forward insuficiente para una decisión dura.")
        flags["insufficient_evidence"] = True

    score = _apply_ratio_penalty(
        score,
        current=_safe_float(forward_metrics.shadow_profit_factor),
        baseline=_safe_float(design_profile.profit_factor_design),
        degraded_floor=float(limits.min_profit_factor_ratio_degraded),
        suspend_floor=float(limits.min_profit_factor_ratio_suspend),
        label="profit_factor",
        reasons=reasons,
        flags=flags,
    )
    score = _apply_ratio_penalty(
        score,
        current=_safe_float(forward_metrics.shadow_sharpe),
        baseline=_safe_float(design_profile.sharpe_design),
        degraded_floor=float(limits.min_sharpe_ratio_degraded),
        suspend_floor=float(limits.min_sharpe_ratio_suspend),
        label="sharpe",
        reasons=reasons,
        flags=flags,
    )

    dd_design = _safe_float(design_profile.max_drawdown_design)
    dd_forward = _safe_float(forward_metrics.shadow_max_drawdown)
    if dd_design is not None and dd_forward is not None:
        flags["drawdown_ratio"] = dd_forward / max(dd_design, 1e-9)
        if dd_forward >= dd_design * float(limits.max_drawdown_multiplier_retire):
            score -= 40.0
            reasons.append("Drawdown forward extremadamente superior al de diseño.")
        elif dd_forward >= dd_design * float(limits.max_drawdown_multiplier_suspend):
            score -= 28.0
            reasons.append("Drawdown forward severamente deteriorado.")
        elif dd_forward >= dd_design * float(limits.max_drawdown_multiplier_degraded):
            score -= 15.0
            reasons.append("Drawdown forward peor que el esperado.")

    avg_loss_design = _safe_float(design_profile.avg_loss_design)
    avg_loss_forward = _safe_float(forward_metrics.shadow_avg_loss)
    if avg_loss_design is not None and avg_loss_forward is not None and avg_loss_design < 0:
        if abs(avg_loss_forward) > abs(avg_loss_design) * 1.25:
            score -= 10.0
            reasons.append("Pérdida media forward peor que la de diseño.")
            flags["avg_loss_ratio"] = abs(avg_loss_forward) / max(abs(avg_loss_design), 1e-9)

    losing_design = _safe_int(design_profile.max_losing_streak_design)
    losing_forward = _safe_int(forward_metrics.shadow_losing_streak)
    if losing_design is not None and losing_forward is not None:
        flags["losing_streak_ratio"] = losing_forward / max(float(losing_design or 1), 1.0)
        if losing_forward > max(int(losing_design), 1) * float(limits.max_losing_streak_multiplier):
            score -= 12.0
            reasons.append("Racha de pérdidas forward peor que la esperada.")

    expectancy_forward = _safe_float(forward_metrics.shadow_expectancy)
    if expectancy_forward is not None and expectancy_forward < 0:
        score -= 10.0
        reasons.append("Expectancy forward negativa.")

    winrate_design = _safe_float(design_profile.winrate_design)
    winrate_forward = _safe_float(forward_metrics.shadow_winrate)
    if winrate_design is not None and winrate_forward is not None and winrate_forward < winrate_design * 0.75:
        score -= 8.0
        reasons.append("Winrate forward significativamente inferior al de diseño.")
        flags["winrate_ratio"] = winrate_forward / max(winrate_design, 1e-9)

    if int(forward_metrics.signal_count or 0) > 0 and int(forward_metrics.ppo_selected_count or 0) == 0:
        score -= 5.0
        reasons.append("El trader genera señales pero no está aportando edge operativo.")

    if int(forward_metrics.ppo_blocked_count or 0) > 0:
        score -= min(10.0, float(forward_metrics.ppo_blocked_count))
        reasons.append("Parte de las señales han sido bloqueadas aguas abajo.")

    score = max(0.0, min(100.0, float(score)))
    previous_state = str(current_state or TraderLifecycleState.LIVE.value)
    new_state = previous_state
    action = RiskAction.KEEP.value
    retrain_request = None

    if bool(flags.get("insufficient_evidence")):
        if score < float(limits.degraded_health_threshold):
            action = RiskAction.DEGRADED.value
            new_state = TraderLifecycleState.DEGRADED.value
        else:
            action = RiskAction.KEEP.value
            new_state = TraderLifecycleState.LIVE.value if previous_state != TraderLifecycleState.SUSPENDED.value else previous_state
    else:
        retire_due_to_drawdown = (
            dd_design is not None
            and dd_forward is not None
            and dd_forward >= dd_design * float(limits.max_drawdown_multiplier_retire)
        )
        retire_due_to_total_breakdown = (
            _safe_float(forward_metrics.shadow_profit_factor) is not None
            and _safe_float(design_profile.profit_factor_design) is not None
            and float(forward_metrics.shadow_profit_factor) < float(design_profile.profit_factor_design) * 0.35
            and _safe_float(forward_metrics.shadow_sharpe) is not None
            and float(forward_metrics.shadow_sharpe) < 0
            and expectancy_forward is not None
            and expectancy_forward < 0
        )
        if (
            shadow_trades >= int(limits.min_forward_trades_for_retire)
            and score < float(limits.retire_health_threshold)
            and (retire_due_to_drawdown or retire_due_to_total_breakdown)
        ):
            action = RiskAction.RETIRE.value
            new_state = TraderLifecycleState.RETIRED.value
            retrain_request = RetrainRequest(
                request_id=f"rr_{uuid4().hex[:10]}",
                trader_id=design_profile.trader_id,
                asset=design_profile.asset,
                timeframe=design_profile.timeframe,
                reason="; ".join(reasons) or "health_score_below_retire_threshold",
                requested_by=AgentKind.RISK,
                context={
                    "health_score": score,
                    "evaluation_run_id": forward_metrics.evaluation_run_id,
                    "evaluation_date": forward_metrics.evaluation_date,
                },
            )
        elif score < float(limits.suspend_health_threshold):
            action = RiskAction.SUSPEND.value
            new_state = TraderLifecycleState.SUSPENDED.value
        elif score < float(limits.degraded_health_threshold):
            action = RiskAction.DEGRADED.value
            new_state = TraderLifecycleState.DEGRADED.value
        else:
            action = RiskAction.KEEP.value
            new_state = TraderLifecycleState.LIVE.value

    if not reasons:
        reasons.append("Trader dentro de los umbrales de salud definidos.")

    return TraderHealthSnapshot(
        trader_id=design_profile.trader_id,
        asset=design_profile.asset,
        timeframe=design_profile.timeframe,
        evaluation_run_id=forward_metrics.evaluation_run_id,
        evaluation_date=forward_metrics.evaluation_date,
        previous_state=previous_state,
        new_state=new_state,
        health_score=score,
        action=action,
        reasons=reasons,
        design_profile=design_profile,
        forward_metrics=forward_metrics,
        risk_flags=flags,
        retrain_request=retrain_request,
        metadata={
            "limits": limits.to_dict(),
            "current_state": previous_state,
            "evaluated_at": utc_now_iso(),
        },
    )
