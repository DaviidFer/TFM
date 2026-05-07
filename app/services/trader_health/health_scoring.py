from __future__ import annotations

from typing import Any, Dict
from uuid import uuid4

from app.contracts import (
    AgentKind,
    RetrainRequest,
    TraderDesignProfile,
    TraderForwardMetrics,
    TraderHealthConfig,
    TraderHealthSnapshot,
    TraderLifecycleState,
    TraderReviewAction,
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
    retraining_floor: float,
    label: str,
    reasons: list[str],
    flags: Dict[str, Any],
    penalty: float = 30.0,
) -> float:
    if baseline is None or current is None:
        return score
    if baseline == 0:
        return score
    ratio = current / baseline
    flags[f"{label}_ratio"] = ratio
    if ratio < retraining_floor:
        reasons.append(f"{label} forward por debajo del umbral admitido.")
        return score - penalty
    return score


def evaluate_trader_health(
    design_profile: TraderDesignProfile,
    forward_metrics: TraderForwardMetrics,
    current_state: str,
    config: TraderHealthConfig,
) -> TraderHealthSnapshot:
    """
    Politica binaria: el trader sigue valido (`KEEP`) o se manda a reentrenamiento
    (`RETRAINING`).

    - Si la evidencia forward es insuficiente (pocos trades), se mantiene LIVE.
    - Si supera los umbrales de salud, se mantiene LIVE.
    - En cuanto cualquier dimension cae bajo el umbral, se manda a RETRAINING
      y se emite un `RetrainRequest`.
    """
    score = 100.0
    reasons: list[str] = []
    flags: Dict[str, Any] = {"insufficient_evidence": bool(forward_metrics.insufficient_evidence)}

    shadow_trades = int(forward_metrics.shadow_trades or 0)
    insufficient = shadow_trades < int(config.min_forward_trades_for_retraining)
    if insufficient:
        flags["insufficient_evidence"] = True
        reasons.append("Evidencia forward insuficiente para una decision dura.")

    score = _apply_ratio_penalty(
        score,
        current=_safe_float(forward_metrics.shadow_profit_factor),
        baseline=_safe_float(design_profile.profit_factor_design),
        retraining_floor=float(config.min_profit_factor_ratio_retraining),
        label="profit_factor",
        reasons=reasons,
        flags=flags,
    )
    score = _apply_ratio_penalty(
        score,
        current=_safe_float(forward_metrics.shadow_sharpe),
        baseline=_safe_float(design_profile.sharpe_design),
        retraining_floor=float(config.min_sharpe_ratio_retraining),
        label="sharpe",
        reasons=reasons,
        flags=flags,
    )

    dd_design = _safe_float(design_profile.max_drawdown_design)
    dd_forward = _safe_float(forward_metrics.shadow_max_drawdown)
    if dd_design is not None and dd_forward is not None:
        flags["drawdown_ratio"] = dd_forward / max(dd_design, 1e-9)
        if dd_forward >= dd_design * float(config.max_drawdown_multiplier_retraining):
            score -= 35.0
            reasons.append("Drawdown forward por encima del umbral admitido.")

    avg_loss_design = _safe_float(design_profile.avg_loss_design)
    avg_loss_forward = _safe_float(forward_metrics.shadow_avg_loss)
    if avg_loss_design is not None and avg_loss_forward is not None and avg_loss_design < 0:
        if abs(avg_loss_forward) > abs(avg_loss_design) * 1.25:
            score -= 10.0
            reasons.append("Perdida media forward peor que la de diseno.")
            flags["avg_loss_ratio"] = abs(avg_loss_forward) / max(abs(avg_loss_design), 1e-9)

    losing_design = _safe_int(design_profile.max_losing_streak_design)
    losing_forward = _safe_int(forward_metrics.shadow_losing_streak)
    if losing_design is not None and losing_forward is not None:
        flags["losing_streak_ratio"] = losing_forward / max(float(losing_design or 1), 1.0)
        if losing_forward > max(int(losing_design), 1) * float(config.max_losing_streak_multiplier):
            score -= 12.0
            reasons.append("Racha de perdidas forward peor que la esperada.")

    expectancy_forward = _safe_float(forward_metrics.shadow_expectancy)
    if expectancy_forward is not None and expectancy_forward < 0:
        score -= 10.0
        reasons.append("Expectancy forward negativa.")

    winrate_design = _safe_float(design_profile.winrate_design)
    winrate_forward = _safe_float(forward_metrics.shadow_winrate)
    if winrate_design is not None and winrate_forward is not None and winrate_forward < winrate_design * 0.75:
        score -= 8.0
        reasons.append("Winrate forward significativamente inferior al de diseno.")
        flags["winrate_ratio"] = winrate_forward / max(winrate_design, 1e-9)

    # Si genera senales pero el PortfolioManagerProcess (GA+PSO) nunca las
    # selecciona, su edge ya no es competitivo dentro de la cartera.
    if int(forward_metrics.signal_count or 0) > 0 and int(forward_metrics.pm_selected_count or 0) == 0:
        score -= 5.0
        reasons.append("El trader genera senales pero nunca lo selecciona el portfolio manager.")

    score = max(0.0, min(100.0, float(score)))
    previous_state = str(current_state or TraderLifecycleState.LIVE.value)

    if insufficient:
        action = TraderReviewAction.KEEP.value
        new_state = TraderLifecycleState.LIVE.value
    elif score < float(config.retraining_health_threshold):
        action = TraderReviewAction.RETRAINING.value
        new_state = TraderLifecycleState.RETRAINING.value
    else:
        action = TraderReviewAction.KEEP.value
        new_state = TraderLifecycleState.LIVE.value

    retrain_request = None
    if action == TraderReviewAction.RETRAINING.value:
        retrain_request = RetrainRequest(
            request_id=f"rr_{uuid4().hex[:10]}",
            trader_id=design_profile.trader_id,
            asset=design_profile.asset,
            timeframe=design_profile.timeframe,
            reason="; ".join(reasons) or "health_score_below_retraining_threshold",
            requested_by=AgentKind.HUMAN_RESOURCES,
            context={
                "health_score": score,
                "evaluation_run_id": forward_metrics.evaluation_run_id,
                "evaluation_date": forward_metrics.evaluation_date,
            },
        )

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
        flags=flags,
        retrain_request=retrain_request,
        metadata={
            "config": config.to_dict(),
            "current_state": previous_state,
            "evaluated_at": utc_now_iso(),
        },
    )
