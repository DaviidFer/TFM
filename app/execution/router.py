from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List
from uuid import uuid4

from app.core.structured_logging import emit_log
from app.execution.access import ensure_execution_access
from app.execution.local_data_provider import LocalMarketDataProvider
from app.execution.models import ExecutionMode, OrderIntent, OrderResult
from app.execution.mt5_connector import MT5Connector


class ExecutionRouter:
    """
    Único punto de acceso a ejecución:
    - PAPER: simula enrutado y mantiene ledger en memoria.
    - LIVE_MT5: enruta con MT5Connector.
    """

    def __init__(
        self,
        *,
        market_data: LocalMarketDataProvider,
        mode: ExecutionMode = ExecutionMode.PAPER,
        mt5_connector: MT5Connector | None = None,
    ) -> None:
        self.market_data = market_data
        self.mode = mode
        self.mt5 = mt5_connector
        self._paper_orders: List[Dict[str, Any]] = []

    def route_order(self, *, actor: str, intent: OrderIntent) -> OrderResult:
        ensure_execution_access(actor)
        if self.mode == ExecutionMode.PAPER:
            ticket = f"paper_{uuid4().hex[:10]}"
            payload = {"ticket": ticket, "intent": asdict(intent)}
            self._paper_orders.append(payload)
            emit_log(
                "execution_router",
                "order_routed_paper",
                console=False,
                actor=actor,
                symbol=intent.symbol,
                side=intent.side.value,
                volume=intent.volume,
                ticket=ticket,
            )
            return OrderResult(
                accepted=True,
                mode=self.mode,
                ticket=ticket,
                reason="paper_order_routed",
                broker_payload=payload,
            )

        if self.mt5 is None:
            return OrderResult(
                accepted=False,
                mode=self.mode,
                ticket="",
                reason="mt5_connector_missing",
            )
        if not self.mt5.connected and not self.mt5.connect():
            return OrderResult(
                accepted=False,
                mode=self.mode,
                ticket="",
                reason="mt5_connect_failed",
            )
        out = self.mt5.send_market_order(intent)
        payload = out.get("payload", out)
        accepted = bool(out.get("ok", False))
        ticket = str(payload.get("order") or payload.get("ticket") or "")
        reason = str(out.get("reason") or ("mt5_order_sent" if accepted else "mt5_order_rejected"))
        emit_log(
            "execution_router",
            "order_routed_mt5",
            console=False,
            actor=actor,
            symbol=intent.symbol,
            side=intent.side.value,
            volume=intent.volume,
            accepted=accepted,
            ticket=ticket,
            reason=reason,
        )
        return OrderResult(
            accepted=accepted,
            mode=self.mode,
            ticket=ticket,
            reason=reason,
            broker_payload=payload if isinstance(payload, dict) else {"payload": str(payload)},
        )

    def close_position(self, *, actor: str, trader_id: str, position: Dict[str, Any], comment: str = "") -> OrderResult:
        ensure_execution_access(actor)
        if self.mode == ExecutionMode.PAPER:
            ticket = f"paper_close_{uuid4().hex[:10]}"
            payload = {"ticket": ticket, "position": dict(position), "trader_id": trader_id}
            emit_log(
                "execution_router",
                "position_closed_paper",
                console=False,
                actor=actor,
                trader_id=trader_id,
                symbol=str(position.get("symbol") or ""),
                ticket=ticket,
            )
            return OrderResult(
                accepted=True,
                mode=self.mode,
                ticket=ticket,
                reason="paper_close_routed",
                broker_payload=payload,
            )

        if self.mt5 is None:
            return OrderResult(
                accepted=False,
                mode=self.mode,
                ticket="",
                reason="mt5_connector_missing",
            )
        if not self.mt5.connected and not self.mt5.connect():
            return OrderResult(
                accepted=False,
                mode=self.mode,
                ticket="",
                reason="mt5_connect_failed",
            )
        out = self.mt5.close_position(position=position, trader_id=trader_id, comment=comment)
        payload = out.get("payload", out)
        accepted = bool(out.get("ok", False))
        ticket = str(payload.get("order") or payload.get("ticket") or position.get("ticket") or "")
        reason = str(out.get("reason") or ("mt5_close_sent" if accepted else "mt5_close_rejected"))
        emit_log(
            "execution_router",
            "position_closed_mt5",
            console=False,
            actor=actor,
            trader_id=trader_id,
            symbol=str(position.get("symbol") or ""),
            accepted=accepted,
            ticket=ticket,
            reason=reason,
        )
        return OrderResult(
            accepted=accepted,
            mode=self.mode,
            ticket=ticket,
            reason=reason,
            broker_payload=payload if isinstance(payload, dict) else {"payload": str(payload)},
        )

    def get_open_positions(self, *, actor: str) -> List[Dict[str, Any]]:
        ensure_execution_access(actor)
        if self.mode == ExecutionMode.PAPER:
            return list(self._paper_orders)
        if self.mt5 is None:
            return []
        return self.mt5.get_open_positions()

    def get_market_snapshot(self, *, actor: str, symbol: str) -> Dict[str, Any]:
        ensure_execution_access(actor)
        latest = self.market_data.get_latest_bar(symbol)
        rng = self.market_data.get_range_info(symbol)
        return {"latest_bar": latest, "range": rng}

    def get_account_info(self, *, actor: str) -> Dict[str, Any]:
        ensure_execution_access(actor)
        if self.mode == ExecutionMode.PAPER:
            return {"mode": self.mode.value, "connected": False, "paper_orders": len(self._paper_orders)}
        if self.mt5 is None:
            return {"mode": self.mode.value, "connected": False}
        return {"mode": self.mode.value, **self.mt5.account_info()}

