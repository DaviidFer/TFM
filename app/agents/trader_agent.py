from __future__ import annotations

from uuid import uuid4

from app.contracts import AgentStatus, EventType, PromotedTraderSpec, TraderLifecycleState, TraderLiveMetrics
from app.core.structured_logging import emit_log
from app.execution import OrderIntent
from app.execution.models import OrderSide

from .base import AgentContext


class TraderAgent:
    agent_id = "trader_agent"

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx

    def activate(self, promoted: PromotedTraderSpec) -> TraderLiveMetrics:
        self.ctx.store.set_agent_status(self.agent_id, AgentStatus.RUNNING, "activating trader")
        emit_log(
            self.agent_id,
            "trader_activation_started",
            trader_id=promoted.trader_id,
            asset=promoted.asset,
            timeframe=promoted.timeframe,
            origin_experiment_id=promoted.origin_experiment_id,
            long_rules=promoted.long_rules,
            short_rules=promoted.short_rules,
        )

        # Fase 4: activación y heartbeat inicial (sin runtime asíncrono todavía).
        n_long = len(promoted.long_rules)
        n_short = len(promoted.short_rules)
        metrics = TraderLiveMetrics(
            trader_id=promoted.trader_id,
            as_of=promoted.promoted_at,
            pnl=0.0,
            sharpe_rolling=0.0,
            drawdown_rolling=0.0,
            trade_count=0,
            extra_metrics={
                "n_long_rules": n_long,
                "n_short_rules": n_short,
                "readiness_score": float(min(1.0, (n_long + n_short) / 20.0)),
            },
        )

        self.ctx.store.upsert_trader_state(
            trader_id=promoted.trader_id,
            asset=promoted.asset,
            timeframe=promoted.timeframe,
            state=TraderLifecycleState.LIVE,
            notes="activated by trader agent",
        )
        self.ctx.store.upsert_trader_metrics(
            trader_id=metrics.trader_id,
            as_of=metrics.as_of,
            pnl=metrics.pnl,
            sharpe_rolling=metrics.sharpe_rolling,
            drawdown_rolling=metrics.drawdown_rolling,
            trade_count=metrics.trade_count,
            extra=metrics.extra_metrics,
        )
        self.ctx.store.append_event(
            event_id=f"evt_{uuid4().hex[:10]}",
            event_type=EventType.TRADER_STATE_CHANGED,
            producer=self.agent_id,
            payload={
                "trader_id": promoted.trader_id,
                "new_state": TraderLifecycleState.LIVE.value,
                "metrics": metrics.to_dict(),
            },
            correlation_id=promoted.origin_experiment_id,
        )
        self.ctx.store.append_event(
            event_id=f"evt_{uuid4().hex[:10]}",
            event_type=EventType.TRADER_METRICS_UPDATED,
            producer=self.agent_id,
            payload=metrics.to_dict(),
            correlation_id=promoted.origin_experiment_id,
        )
        emit_log(
            self.agent_id,
            "trader_live",
            trader_id=metrics.trader_id,
            metrics=metrics.to_dict(),
        )
        self.ctx.store.set_agent_status(self.agent_id, AgentStatus.IDLE, "trader live")
        return metrics

    def publish_metrics(self, metrics: TraderLiveMetrics, *, correlation_id: str | None = None) -> None:
        self.ctx.store.upsert_trader_metrics(
            trader_id=metrics.trader_id,
            as_of=metrics.as_of,
            pnl=metrics.pnl,
            sharpe_rolling=metrics.sharpe_rolling,
            drawdown_rolling=metrics.drawdown_rolling,
            trade_count=metrics.trade_count,
            extra=metrics.extra_metrics,
        )
        self.ctx.store.append_event(
            event_id=f"evt_{uuid4().hex[:10]}",
            event_type=EventType.TRADER_METRICS_UPDATED,
            producer=self.agent_id,
            payload=metrics.to_dict(),
            correlation_id=correlation_id,
        )
        emit_log(
            self.agent_id,
            "trader_metrics_published",
            trader_id=metrics.trader_id,
            correlation_id=correlation_id,
            metrics=metrics.to_dict(),
        )

    def route_order(
        self,
        *,
        trader_id: str,
        symbol: str,
        side: str,
        volume: float,
        sl: float | None = None,
        tp: float | None = None,
        comment: str = "",
        correlation_id: str | None = None,
    ) -> dict:
        if self.ctx.execution_router is None:
            return {"accepted": False, "reason": "execution_router_not_configured"}
        mapped_side = OrderSide.BUY if str(side).lower() == "buy" else OrderSide.SELL
        intent = OrderIntent(
            trader_id=trader_id,
            symbol=symbol.upper(),
            side=mapped_side,
            volume=float(volume),
            sl=sl,
            tp=tp,
            comment=comment or f"trader:{trader_id}",
        )
        try:
            result = self.ctx.execution_router.route_order(actor=self.agent_id, intent=intent)
        except PermissionError as exc:
            self.ctx.store.append_event(
                event_id=f"evt_{uuid4().hex[:10]}",
                event_type=EventType.BROKER_ACCESS_DENIED,
                producer=self.agent_id,
                payload={"trader_id": trader_id, "symbol": symbol, "error": str(exc)},
                correlation_id=correlation_id,
            )
            return {"accepted": False, "reason": "access_denied", "error": str(exc)}

        event_type = EventType.BROKER_ORDER_ROUTED if result.accepted else EventType.BROKER_ORDER_REJECTED
        payload = {
            "trader_id": trader_id,
            "symbol": symbol.upper(),
            "side": mapped_side.value,
            "volume": float(volume),
            "result": result.__dict__,
        }
        self.ctx.store.append_event(
            event_id=f"evt_{uuid4().hex[:10]}",
            event_type=event_type,
            producer=self.agent_id,
            payload=payload,
            correlation_id=correlation_id,
        )
        emit_log(
            self.agent_id,
            "broker_order_attempt",
            console=False,
            trader_id=trader_id,
            symbol=symbol.upper(),
            side=mapped_side.value,
            volume=float(volume),
            accepted=result.accepted,
            mode=result.mode.value,
            ticket=result.ticket,
            reason=result.reason,
        )
        return result.__dict__

    def close_position(
        self,
        *,
        trader_id: str,
        position: dict,
        correlation_id: str | None = None,
        comment: str = "",
    ) -> dict:
        if self.ctx.execution_router is None:
            return {"accepted": False, "reason": "execution_router_not_configured"}
        try:
            result = self.ctx.execution_router.close_position(
                actor=self.agent_id,
                trader_id=trader_id,
                position=position,
                comment=comment or f"close:{trader_id}",
            )
        except PermissionError as exc:
            self.ctx.store.append_event(
                event_id=f"evt_{uuid4().hex[:10]}",
                event_type=EventType.BROKER_ACCESS_DENIED,
                producer=self.agent_id,
                payload={"trader_id": trader_id, "position": position, "error": str(exc)},
                correlation_id=correlation_id,
            )
            return {"accepted": False, "reason": "access_denied", "error": str(exc)}

        event_type = EventType.BROKER_ORDER_ROUTED if result.accepted else EventType.BROKER_ORDER_REJECTED
        payload = {
            "trader_id": trader_id,
            "action": "close_position",
            "position": dict(position),
            "result": result.__dict__,
        }
        self.ctx.store.append_event(
            event_id=f"evt_{uuid4().hex[:10]}",
            event_type=event_type,
            producer=self.agent_id,
            payload=payload,
            correlation_id=correlation_id,
        )
        emit_log(
            self.agent_id,
            "broker_close_attempt",
            console=False,
            trader_id=trader_id,
            symbol=str(position.get("symbol") or ""),
            accepted=result.accepted,
            mode=result.mode.value,
            ticket=result.ticket,
            reason=result.reason,
        )
        return result.__dict__

