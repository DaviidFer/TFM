from __future__ import annotations

from datetime import datetime, timedelta
from queue import Empty, Queue
from time import sleep
from typing import Callable, Dict, Iterable, Mapping, Any

import pandas as pd

from app.agents import PortfolioManagerProcess, TraderAgent
from app.contracts import PromotedTraderSpec
from app.core.structured_logging import emit_log
from app.execution.mt5_events import DataEvent
from app.services import build_features


class LiveTradingRuntime:
    """
    Runtime D1 que reutiliza la logica de data_provider del mt5-framework:
    - espera nueva vela cerrada por simbolo,
    - recalcula features,
    - evalua reglas del trader,
    - enruta orden automaticamente si hay senal.

    El runtime ya no integra ningun gate previo a ejecucion: la decision del
    `PortfolioManagerProcess` se aplica tal cual. La supervision de salud de
    los traders la hace `HumanResourcesProcess` de forma asincrona, fuera del
    camino caliente de operativa.
    """

    def __init__(
        self,
        *,
        trader_agent: TraderAgent,
        portfolio_manager: PortfolioManagerProcess | None = None,
        promoted_specs: Mapping[str, PromotedTraderSpec],
        data_provider: Any,
        history_loader: Callable[[str], pd.DataFrame | pd.Series | None] | None = None,
        capital_provider: Callable[[], float] | None = None,
        universe_ready_provider: Callable[[], bool] | None = None,
        timeframe: str = "1d",
        bars_lookback: int = 260,
    ) -> None:
        self.trader_agent = trader_agent
        self.portfolio_manager = portfolio_manager
        self.promoted_specs = dict(promoted_specs)
        self.history_loader = history_loader
        self.capital_provider = capital_provider
        self.universe_ready_provider = universe_ready_provider
        self.bars_lookback = bars_lookback
        self.timeframe = timeframe
        self.data_provider = data_provider
        self._queue: Queue = getattr(self.data_provider, "events_queue")
        self._traders_by_symbol: Dict[str, list[PromotedTraderSpec]] = {}
        self._pending_orders: Dict[str, Dict[str, Any]] = {}
        self._pending_closures: Dict[str, Dict[str, Any]] = {}
        self._signal_book: Dict[str, Dict[str, Any]] = {}
        self._last_portfolio_output: Dict[str, Any] | None = None
        self._retry_interval = timedelta(minutes=30)
        self._default_volume = 1.0
        self._last_rebalance_week: str | None = None
        self._initial_portfolio_deployment_done = False
        for spec in promoted_specs.values():
            self._traders_by_symbol.setdefault(spec.asset, []).append(spec)
        if self.portfolio_manager is not None and hasattr(self.portfolio_manager, "sync_universe"):
            try:
                self.portfolio_manager.sync_universe(self.promoted_specs)
            except Exception:
                pass
        self._load_pending_orders_from_store()

    def _current_portfolio_phase(self) -> str:
        return "rebalanceo_semanal" if self._initial_portfolio_deployment_done else "despliegue_inicial"

    def upsert_trader(self, spec: PromotedTraderSpec) -> None:
        self.promoted_specs[spec.trader_id] = spec
        bucket = self._traders_by_symbol.setdefault(spec.asset, [])
        if all(t.trader_id != spec.trader_id for t in bucket):
            bucket.append(spec)
        if self.portfolio_manager is not None and hasattr(self.portfolio_manager, "sync_universe"):
            try:
                self.portfolio_manager.sync_universe(self.promoted_specs)
            except Exception:
                pass
        # Si el data provider soporta universo dinámico, añadimos símbolo.
        symbols = getattr(self.data_provider, "symbols", None)
        if isinstance(symbols, list) and spec.asset not in symbols:
            symbols.append(spec.asset)
            last_bar = getattr(self.data_provider, "last_bar_datetime", None)
            if isinstance(last_bar, dict) and spec.asset not in last_bar:
                from datetime import datetime

                last_bar[spec.asset] = datetime.min

    def stop(self) -> None:
        self._pending_orders.clear()
        self._pending_closures.clear()

    def bootstrap_now(self) -> int:
        """
        Fuerza un barrido inicial de datos al arrancar la operativa para
        evaluar señales inmediatamente sin esperar a una nueva vela.
        """
        try:
            self.data_provider.check_for_new_data(force_emit_snapshot=True)
        except TypeError:
            self.data_provider.check_for_new_data()
        processed = 0
        signal_candidates: list[Dict[str, Any]] = []
        close_candidates: list[Dict[str, Any]] = []
        open_positions_by_trader = self._open_positions_by_trader()
        while True:
            try:
                event = self._queue.get(block=False)
            except Empty:
                break
            if isinstance(event, DataEvent):
                processed += 1
                event_signals, event_closes = self._process_data_event(event, open_positions_by_trader)
                signal_candidates.extend(event_signals)
                close_candidates.extend(event_closes)
        for candidate in close_candidates:
            self._attempt_close(
                spec=candidate["spec"],
                position=candidate["position"],
                source="signal",
                rationale=str(candidate.get("reason") or "signal_inactive"),
            )
        if signal_candidates:
            self._process_signal_candidates(signal_candidates)
        return processed

    def get_pending_orders(self) -> list[Dict[str, Any]]:
        out: list[Dict[str, Any]] = []
        for key, payload in self._pending_orders.items():
            row = dict(payload)
            row["pending_key"] = key
            nra = row.get("next_retry_at")
            if isinstance(nra, datetime):
                row["next_retry_at"] = nra.isoformat()
            out.append(row)
        out.sort(key=lambda x: str(x.get("next_retry_at", "")))
        return out

    def get_signal_book(self) -> list[Dict[str, Any]]:
        rows = [dict(v) for v in self._signal_book.values()]
        rows.sort(key=lambda x: str(x.get("detected_at", "")), reverse=True)
        return rows

    def get_last_portfolio_output(self) -> Dict[str, Any] | None:
        return self._last_portfolio_output

    def _load_pending_orders_from_store(self) -> None:
        try:
            rows = self.trader_agent.ctx.store.list_pending_orders()
        except Exception:
            return
        for row in rows:
            try:
                next_retry_at = datetime.fromisoformat(str(row.get("next_retry_at")).replace("Z", "+00:00"))
            except Exception:
                next_retry_at = datetime.now()
            key = str(row.get("pending_key"))
            self._pending_orders[key] = {
                "trader_id": str(row.get("trader_id") or ""),
                "symbol": str(row.get("symbol") or "").upper(),
                "side": str(row.get("side") or "").lower(),
                "volume": float(row.get("volume") or self._default_volume),
                "correlation_id": row.get("correlation_id"),
                "signal_label": str(row.get("signal_label") or ""),
                "attempts": int(row.get("attempts") or 0),
                "next_retry_at": next_retry_at,
                "last_reason": str(row.get("last_reason") or ""),
            }

    def _evaluate_rules(self, row_df: pd.DataFrame, rules: Iterable[str]) -> bool:
        for rule in rules:
            try:
                out = row_df.eval(str(rule), engine="python")
                if bool(out.iloc[-1]):
                    return True
            except Exception:
                continue
        return False

    def _dateprint(self) -> str:
        return datetime.now().strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]

    def _trader_magic(self, trader_id: str) -> int:
        txt = str(trader_id or "")
        digits = "".join(ch for ch in txt if ch.isdigit())
        if digits:
            try:
                return int(digits[-9:])
            except Exception:
                pass
        return abs(hash(txt)) % 900_000_000 + 100_000_000

    def _position_direction(self, position: Dict[str, Any]) -> str:
        pos_type = int(position.get("type", -1))
        if pos_type == 0:
            return "buy"
        if pos_type == 1:
            return "sell"
        return ""

    def _week_token(self, when: datetime | None = None) -> str:
        ref = when or datetime.now()
        iso = ref.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"

    def _can_open_new_positions(self) -> bool:
        if self.universe_ready_provider is None:
            return True
        try:
            return bool(self.universe_ready_provider())
        except Exception:
            return False

    def _is_rebalance_day(self, when: datetime | None = None) -> bool:
        ref = when or datetime.now()
        return int(ref.weekday()) == 0

    def _should_run_weekly_rebalance(self, when: datetime | None = None) -> bool:
        if not self._initial_portfolio_deployment_done:
            return True
        ref = when or datetime.now()
        if not self._is_rebalance_day(ref):
            return False
        week_token = self._week_token(ref)
        return week_token != self._last_rebalance_week

    def _mark_rebalance_executed(self, when: datetime | None = None) -> None:
        self._initial_portfolio_deployment_done = True
        self._last_rebalance_week = self._week_token(when or datetime.now())

    def _get_open_positions(self) -> list[Dict[str, Any]]:
        try:
            return list(self.trader_agent.ctx.execution_router.get_open_positions(actor=self.trader_agent.agent_id))
        except Exception:
            return []

    def _open_positions_by_trader(self) -> Dict[str, list[Dict[str, Any]]]:
        mapping: Dict[str, list[Dict[str, Any]]] = {}
        by_magic: Dict[int, str] = {self._trader_magic(tid): tid for tid in self.promoted_specs}
        for position in self._get_open_positions():
            trader_id = by_magic.get(int(position.get("magic", -1)))
            if trader_id is None:
                continue
            mapping.setdefault(trader_id, []).append(dict(position))
        return mapping

    def _tf_label(self) -> str:
        tf = str(getattr(self.data_provider, "timeframe", self.timeframe)).lower()
        mapping = {"1d": "D1", "1h": "H1", "4h": "H4", "1w": "W1", "1m": "MN1"}
        return mapping.get(tf, tf.upper())

    def _build_feature_row(self, symbol: str) -> pd.DataFrame:
        bars = self.data_provider.get_latest_closed_bars(symbol, timeframe=self.data_provider.timeframe, num_bars=self.bars_lookback)
        if bars.empty:
            return pd.DataFrame()
        ohlc = bars[["open", "high", "low", "close"]].copy()
        features = build_features(ohlc, dropna=True)
        if features.empty:
            return pd.DataFrame()
        return features.tail(1)

    def _pending_key(self, *, trader_id: str, symbol: str, side: str) -> str:
        return f"{trader_id}|{symbol.upper()}|{side.lower()}"

    def _schedule_retry(
        self,
        *,
        spec: PromotedTraderSpec,
        symbol: str,
        side: str,
        volume: float,
        signal_label: str,
        previous_attempts: int = 0,
    ) -> None:
        now = datetime.now()
        key = self._pending_key(trader_id=spec.trader_id, symbol=symbol, side=side)
        attempts = int(previous_attempts) + 1
        next_retry_at = now + self._retry_interval
        self._pending_orders[key] = {
            "trader_id": spec.trader_id,
            "symbol": symbol.upper(),
            "side": side.lower(),
            "volume": float(max(self._default_volume, volume)),
            "correlation_id": spec.origin_experiment_id,
            "signal_label": signal_label,
            "attempts": attempts,
            "next_retry_at": next_retry_at,
            "last_reason": "",
        }
        try:
            self.trader_agent.ctx.store.upsert_pending_order(
                pending_key=key,
                trader_id=spec.trader_id,
                symbol=symbol.upper(),
                side=side.lower(),
                volume=float(max(self._default_volume, volume)),
                signal_label=signal_label,
                correlation_id=spec.origin_experiment_id,
                attempts=attempts,
                next_retry_at=next_retry_at.isoformat(),
                last_reason="",
            )
        except Exception:
            pass
        print(
            f"{self._dateprint()} - AVISO: reintento programado para {symbol.upper()} "
            f"({signal_label}) en 30 minutos. Intentos fallidos={attempts}"
        )

    def _closure_key(self, *, trader_id: str, ticket: Any) -> str:
        return f"{trader_id}|close|{ticket}"

    def _schedule_close_retry(
        self,
        *,
        spec: PromotedTraderSpec,
        position: Dict[str, Any],
        signal_label: str,
        previous_attempts: int = 0,
    ) -> None:
        ticket = position.get("ticket") or position.get("identifier") or ""
        key = self._closure_key(trader_id=spec.trader_id, ticket=ticket)
        attempts = int(previous_attempts) + 1
        self._pending_closures[key] = {
            "trader_id": spec.trader_id,
            "position": dict(position),
            "signal_label": signal_label,
            "attempts": attempts,
            "next_retry_at": datetime.now() + self._retry_interval,
            "correlation_id": spec.origin_experiment_id,
        }
        print(
            f"{self._dateprint()} - AVISO: reintento de cierre programado para {spec.asset.upper()} "
            f"({signal_label}) en 30 minutos. Intentos fallidos={attempts}"
        )

    def _print_rejection_details(self, symbol: str, signal_label: str, res: Dict[str, Any]) -> None:
        broker_payload = res.get("broker_payload") if isinstance(res.get("broker_payload"), dict) else {}
        retcode = broker_payload.get("retcode")
        retcode_name = broker_payload.get("retcode_name")
        comment = broker_payload.get("comment")
        retcode_external = broker_payload.get("retcode_external")
        last_error = broker_payload.get("last_error")
        print(
            f"{self._dateprint()} - AVISO: orden rechazada para {symbol} "
            f"({signal_label}) motivo={res.get('reason')} "
            f"retcode={retcode} retcode_name={retcode_name} "
            f"comment={comment} retcode_external={retcode_external} last_error={last_error}"
        )

    def _signal_book_key(self, trader_id: str, symbol: str, side: str) -> str:
        return f"{trader_id}|{symbol.upper()}|{side.lower()}"

    def _update_signal_book(self, payload: Dict[str, Any]) -> None:
        key = self._signal_book_key(str(payload.get("trader_id")), str(payload.get("symbol")), str(payload.get("side")))
        merged = dict(self._signal_book.get(key, {}))
        merged.update(payload)
        self._signal_book[key] = merged

    def _current_total_capital(self) -> float:
        if self.capital_provider is not None:
            try:
                capital = float(self.capital_provider())
                if capital > 0:
                    return capital
            except Exception:
                pass
        return 100000.0

    def _estimate_volume_from_euros(self, *, price: float, euros: float) -> float:
        if price <= 0:
            return self._default_volume
        units = float(euros) / float(price)
        units_int = int(max(self._default_volume, units))
        return float(max(self._default_volume, units_int))

    def _audit_signal(
        self,
        *,
        trader_id: str,
        asset: str,
        timeframe: str,
        signal_side: str,
        signal_active: bool,
        pm_selected: bool,
        pm_weight: float,
        executed: bool,
        reason_if_blocked: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            self.trader_agent.ctx.store.save_trader_signal_audit(
                timestamp=datetime.now().isoformat(),
                trader_id=trader_id,
                asset=asset,
                timeframe=timeframe,
                signal_side=signal_side,
                signal_active=signal_active,
                pm_selected=pm_selected,
                pm_weight=float(pm_weight),
                executed=executed,
                reason_if_blocked=reason_if_blocked,
                metadata=dict(metadata or {}),
            )
        except Exception:
            pass

    def _attempt_order(
        self,
        *,
        spec: PromotedTraderSpec,
        symbol: str,
        side: str,
        volume: float,
        signal_label: str,
        source: str,
        previous_attempts: int = 0,
        portfolio_weight: float | None = None,
        portfolio_euros: float | None = None,
        rebalance_id: str = "",
        detected_at: str = "",
    ) -> Dict[str, Any]:
        tf_label = self._tf_label()
        v = float(max(self._default_volume, volume))
        if source == "retry":
            print(
                f"{self._dateprint()} - Recibido RETRY ORDER EVENT con volumen {v} "
                f"para {signal_label} en {symbol} ({tf_label})"
            )
        else:
            print(
                f"{self._dateprint()} - Recibido SIZING EVENT con volumen {v} "
                f"para {signal_label} en {symbol} ({tf_label})"
            )
            print(
                f"{self._dateprint()} - Recibido ORDER EVENT con volumen {v} "
                f"para {signal_label} en {symbol} ({tf_label})"
            )

        res = self.trader_agent.route_order(
            trader_id=spec.trader_id,
            symbol=symbol,
            side=side,
            volume=v,
            comment=f"live_runtime_d1:{spec.trader_id}",
            correlation_id=spec.origin_experiment_id,
        )
        key = self._pending_key(trader_id=spec.trader_id, symbol=symbol, side=side)
        if bool(res.get("accepted")):
            print(
                f"{self._dateprint()} - La Market Order {signal_label} para {symbol} "
                f"de {v} unidades se ha ejecutado correctamente."
            )
            exec_price = 0.0
            broker_payload = res.get("broker_payload")
            if isinstance(broker_payload, dict):
                if "price" in broker_payload:
                    exec_price = float(broker_payload.get("price") or 0.0)
                elif isinstance(broker_payload.get("payload"), dict):
                    exec_price = float(broker_payload["payload"].get("price") or 0.0)
            print(
                f"{self._dateprint()} - Recibido EXECUTION EVENT {signal_label} "
                f"en {symbol} ({tf_label}) con volumen {v} a precio {exec_price}"
            )
            self._pending_orders.pop(key, None)
            try:
                self.trader_agent.ctx.store.delete_pending_order(key)
            except Exception:
                pass
            self._update_signal_book(
                {
                    "trader_id": spec.trader_id,
                    "symbol": symbol.upper(),
                    "side": side.lower(),
                    "signal_label": signal_label,
                    "status": "executed",
                    "selected": True,
                    "weight": float(portfolio_weight or 0.0),
                    "euros": float(portfolio_euros or 0.0),
                    "volume": float(v),
                    "ticket": res.get("ticket"),
                    "reason": res.get("reason"),
                    "detected_at": str(detected_at or datetime.now().isoformat()),
                    "portfolio_phase": self._current_portfolio_phase(),
                    "rebalance_id": str(rebalance_id or ""),
                }
            )
        else:
            self._print_rejection_details(symbol, signal_label, res)
            self._schedule_retry(
                spec=spec,
                symbol=symbol,
                side=side,
                volume=v,
                signal_label=signal_label,
                previous_attempts=previous_attempts,
            )
            pending = self._pending_orders.get(key, {})
            pending["last_reason"] = str(res.get("reason") or "")
            self._pending_orders[key] = pending
            try:
                self.trader_agent.ctx.store.upsert_pending_order(
                    pending_key=key,
                    trader_id=spec.trader_id,
                    symbol=symbol.upper(),
                    side=side.lower(),
                    volume=v,
                    signal_label=signal_label,
                    correlation_id=spec.origin_experiment_id,
                    attempts=int(pending.get("attempts") or 1),
                    next_retry_at=(pending.get("next_retry_at") or datetime.now()).isoformat() if isinstance(pending.get("next_retry_at"), datetime) else str(pending.get("next_retry_at")),
                    last_reason=str(res.get("reason") or ""),
                )
            except Exception:
                pass
            self._update_signal_book(
                {
                    "trader_id": spec.trader_id,
                    "symbol": symbol.upper(),
                    "side": side.lower(),
                    "signal_label": signal_label,
                    "status": "rejected",
                    "selected": True,
                    "weight": float(portfolio_weight or 0.0),
                    "euros": float(portfolio_euros or 0.0),
                    "volume": float(v),
                    "ticket": res.get("ticket"),
                    "reason": res.get("reason"),
                    "detected_at": str(detected_at or datetime.now().isoformat()),
                    "portfolio_phase": self._current_portfolio_phase(),
                    "rebalance_id": str(rebalance_id or ""),
                }
            )

        emit_log(
            "live_runtime",
            "signal_executed",
            console=False,
            trader_id=spec.trader_id,
            symbol=symbol,
            side=side,
            source=source,
            accepted=bool(res.get("accepted")),
            reason=res.get("reason"),
        )
        self._audit_signal(
            trader_id=str(spec.trader_id),
            asset=str(symbol).upper(),
            timeframe=self._tf_label(),
            signal_side=str(side).lower(),
            signal_active=True,
            pm_selected=True,
            pm_weight=float(portfolio_weight or 0.0),
            executed=bool(res.get("accepted")),
            reason_if_blocked="" if bool(res.get("accepted")) else str(res.get("reason") or ""),
            metadata={
                "signal_label": signal_label,
                "source": source,
                "portfolio_euros": float(portfolio_euros or 0.0),
                "volume": float(v),
                "ticket": res.get("ticket"),
                "rebalance_id": str(rebalance_id or ""),
                "detected_at": str(detected_at or ""),
            },
        )
        return res

    def _attempt_close(
        self,
        *,
        spec: PromotedTraderSpec,
        position: Dict[str, Any],
        source: str,
        previous_attempts: int = 0,
        rationale: str = "signal_inactive",
    ) -> Dict[str, Any]:
        symbol = str(position.get("symbol") or spec.asset).upper()
        volume = float(position.get("volume") or self._default_volume)
        signal_label = f"CLOSE_{self._position_direction(position).upper() or 'POSITION'}"
        if source == "retry":
            print(
                f"{self._dateprint()} - Recibido RETRY CLOSE EVENT para {symbol} "
                f"con volumen {volume}"
            )
        else:
            print(
                f"{self._dateprint()} - Recibido CLOSE EVENT para {symbol} "
                f"con volumen {volume} motivo={rationale}"
            )

        res = self.trader_agent.close_position(
            trader_id=spec.trader_id,
            position=position,
            correlation_id=spec.origin_experiment_id,
            comment=f"close_runtime:{spec.trader_id}",
        )
        key = self._closure_key(trader_id=spec.trader_id, ticket=position.get("ticket") or position.get("identifier") or "")
        if bool(res.get("accepted")):
            print(
                f"{self._dateprint()} - Cierre ejecutado correctamente en {symbol} "
                f"para trader {spec.trader_id}."
            )
            self._pending_closures.pop(key, None)
            self._update_signal_book(
                {
                    "trader_id": spec.trader_id,
                    "symbol": symbol,
                    "side": self._position_direction(position),
                    "signal_label": signal_label,
                    "status": "closed",
                    "selected": False,
                    "volume": volume,
                    "reason": rationale,
                    "detected_at": datetime.now().isoformat(),
                    "portfolio_phase": self._current_portfolio_phase(),
                }
            )
        else:
            self._print_rejection_details(symbol, signal_label, res)
            self._schedule_close_retry(
                spec=spec,
                position=position,
                signal_label=signal_label,
                previous_attempts=previous_attempts,
            )
            self._update_signal_book(
                {
                    "trader_id": spec.trader_id,
                    "symbol": symbol,
                    "side": self._position_direction(position),
                    "signal_label": signal_label,
                    "status": "close_rejected",
                    "selected": False,
                    "volume": volume,
                    "reason": str(res.get("reason") or rationale),
                    "detected_at": datetime.now().isoformat(),
                    "portfolio_phase": self._current_portfolio_phase(),
                }
            )
        return res

    def _process_signal_candidates(
        self,
        signal_candidates: list[Dict[str, Any]],
        *,
        force_rebalance: bool = False,
        manual_reason: str = "",
    ) -> None:
        if not signal_candidates:
            self._last_portfolio_output = None
            return

        if not self._can_open_new_positions():
            self._last_portfolio_output = {
                "selected_tickers": [],
                "weights": {},
                "euros": {},
                "comparison": pd.DataFrame(),
                "figures": {},
                "status": "waiting_full_universe",
                "portfolio_phase": self._current_portfolio_phase(),
            }
            for candidate in signal_candidates:
                self._update_signal_book(
                    {
                        "trader_id": candidate["trader_id"],
                        "symbol": candidate["symbol"],
                        "side": candidate["side"],
                        "signal_label": candidate["signal_label"],
                        "status": "waiting_full_universe",
                        "selected": False,
                        "weight": 0.0,
                        "euros": 0.0,
                        "price": float(candidate.get("price") or 0.0),
                        "detected_at": candidate["detected_at"],
                        "reason": "portfolio_manager_waiting_full_universe",
                        "portfolio_phase": self._current_portfolio_phase(),
                    }
                )
            return

        if (not force_rebalance) and (not self._should_run_weekly_rebalance()):
            self._last_portfolio_output = {
                "selected_tickers": [],
                "weights": {},
                "euros": {},
                "comparison": pd.DataFrame(),
                "figures": {},
                "status": "waiting_next_monday",
                "portfolio_phase": self._current_portfolio_phase(),
            }
            for candidate in signal_candidates:
                self._update_signal_book(
                    {
                        "trader_id": candidate["trader_id"],
                        "symbol": candidate["symbol"],
                        "side": candidate["side"],
                        "signal_label": candidate["signal_label"],
                        "status": "waiting_next_monday",
                        "selected": False,
                        "weight": 0.0,
                        "euros": 0.0,
                        "price": float(candidate.get("price") or 0.0),
                        "detected_at": candidate["detected_at"],
                        "reason": "portfolio_manager_weekly_rebalance_only",
                        "portfolio_phase": self._current_portfolio_phase(),
                    }
                )
            return

        total_capital = self._current_total_capital()
        active_signals = [
            {
                "trader_id": s["trader_id"],
                "symbol": s["symbol"],
                "side": s["side"],
                "signal_label": s["signal_label"],
                "price": s.get("price"),
            }
            for s in signal_candidates
        ]

        if self.portfolio_manager is not None:
            try:
                pm_out = self.portfolio_manager.rebalance_active_signals(
                    active_signals=active_signals,
                    total_capital_eur=total_capital,
                    history_loader=self.history_loader,
                    frequency="daily",
                    lookback=252,
                )
            except Exception as exc:
                emit_log("live_runtime", "portfolio_manager_error", console=False, error=str(exc))
                pm_out = {
                    "selected_tickers": [s["trader_id"] for s in signal_candidates],
                    "weights": {s["trader_id"]: 1.0 / len(signal_candidates) for s in signal_candidates},
                    "euros": {s["trader_id"]: total_capital / len(signal_candidates) for s in signal_candidates},
                    "comparison": pd.DataFrame(),
                    "figures": {},
                }
        else:
            pm_out = {
                "selected_tickers": [s["trader_id"] for s in signal_candidates],
                "weights": {s["trader_id"]: 1.0 / len(signal_candidates) for s in signal_candidates},
                "euros": {s["trader_id"]: total_capital / len(signal_candidates) for s in signal_candidates},
                "comparison": pd.DataFrame(),
                "figures": {},
            }

        pm_out["portfolio_phase"] = self._current_portfolio_phase()
        if force_rebalance:
            pm_out["status"] = "manual_rebalance_executed"
            pm_out["manual_retrain_reason"] = str(manual_reason or "manual_ui_retrain")
        decision_payload = dict(pm_out.get("decision") or {})
        rebalance_id = str(decision_payload.get("decision_id") or "")
        selected_ids = set(pm_out.get("selected_tickers", []) or [])
        weights = dict(pm_out.get("weights", {}) or {})
        euros_map = dict(pm_out.get("euros", {}) or {})
        self._last_portfolio_output = pm_out

        for candidate in signal_candidates:
            trader_id = str(candidate["trader_id"])
            selected = trader_id in selected_ids
            already_open = bool(candidate.get("already_open"))
            self._update_signal_book(
                {
                    "trader_id": trader_id,
                    "symbol": candidate["symbol"],
                    "side": candidate["side"],
                    "signal_label": candidate["signal_label"],
                    "status": ("already_open" if (selected and already_open) else ("selected" if selected else "discarded")),
                    "selected": bool(selected),
                    "weight": float(weights.get(trader_id, 0.0)),
                    "euros": float(euros_map.get(trader_id, 0.0)),
                    "price": float(candidate.get("price") or 0.0),
                    "detected_at": candidate["detected_at"],
                    "reason": str(candidate.get("reason") or ("position_already_open" if already_open else manual_reason or "")),
                    "portfolio_phase": self._current_portfolio_phase(),
                    "rebalance_id": rebalance_id,
                }
            )
            self._audit_signal(
                trader_id=trader_id,
                asset=str(candidate["symbol"]).upper(),
                timeframe=self._tf_label(),
                signal_side=str(candidate["side"]),
                signal_active=True,
                pm_selected=bool(selected),
                pm_weight=float(weights.get(trader_id, 0.0)),
                executed=False,
                reason_if_blocked="",
                metadata={
                    "signal_label": candidate["signal_label"],
                    "portfolio_phase": self._current_portfolio_phase(),
                    "rebalance_id": rebalance_id,
                },
            )

        for candidate in signal_candidates:
            trader_id = str(candidate["trader_id"])
            already_open = bool(candidate.get("already_open"))
            if trader_id not in selected_ids:
                if force_rebalance and already_open:
                    for position in list(candidate.get("open_positions") or []):
                        self._attempt_close(
                            spec=candidate["spec"],
                            position=dict(position),
                            source="signal",
                            rationale="portfolio_manual_rebalance_deselected",
                        )
                continue
            if already_open:
                continue
            euros = float(euros_map.get(trader_id, 0.0))
            price = float(candidate.get("price") or 0.0)
            volume = self._estimate_volume_from_euros(price=price, euros=euros)
            self._attempt_order(
                spec=candidate["spec"],
                symbol=candidate["symbol"],
                side=candidate["side"],
                volume=volume,
                signal_label=candidate["signal_label"],
                source="signal",
                portfolio_weight=float(weights.get(trader_id, 0.0)),
                portfolio_euros=euros,
                rebalance_id=rebalance_id,
                detected_at=str(candidate.get("detected_at") or ""),
            )
        self._mark_rebalance_executed()

    def _process_pending_retries(self) -> None:
        if not self._pending_orders:
            pass
        now = datetime.now()
        keys = list(self._pending_orders.keys())
        for key in keys:
            pending = self._pending_orders.get(key)
            if not pending:
                continue
            next_retry_at = pending.get("next_retry_at")
            if not isinstance(next_retry_at, datetime) or now < next_retry_at:
                continue
            trader_id = str(pending.get("trader_id") or "")
            spec = self.promoted_specs.get(trader_id)
            if spec is None:
                self._pending_orders.pop(key, None)
                continue
            self._attempt_order(
                spec=spec,
                symbol=str(pending.get("symbol") or spec.asset),
                side=str(pending.get("side") or "buy"),
                volume=float(pending.get("volume") or self._default_volume),
                signal_label=str(
                    pending.get("signal_label")
                    or ("SignalType.BUY" if str(pending.get("side")).lower() == "buy" else "SignalType.SELL")
                ),
                source="retry",
                previous_attempts=int(pending.get("attempts") or 0),
            )
        if not self._pending_closures:
            return
        close_keys = list(self._pending_closures.keys())
        for key in close_keys:
            pending = self._pending_closures.get(key)
            if not pending:
                continue
            next_retry_at = pending.get("next_retry_at")
            if not isinstance(next_retry_at, datetime) or now < next_retry_at:
                continue
            trader_id = str(pending.get("trader_id") or "")
            spec = self.promoted_specs.get(trader_id)
            if spec is None:
                self._pending_closures.pop(key, None)
                continue
            self._attempt_close(
                spec=spec,
                position=dict(pending.get("position") or {}),
                source="retry",
                previous_attempts=int(pending.get("attempts") or 0),
                rationale="signal_inactive",
            )

    def _process_data_event(
        self,
        event: DataEvent,
        open_positions_by_trader: Dict[str, list[Dict[str, Any]]],
    ) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
        symbol = event.symbol
        close_px = float(event.data.get("close", 0.0)) if hasattr(event.data, "get") else 0.0
        print(f"{self._dateprint()} - Recibido DATA EVENT de {symbol} - Último precio de cierre: {close_px}")
        emit_log("live_runtime", "data_event_received", console=False, symbol=symbol, bar_time=str(event.data.name))
        feature_row = self._build_feature_row(symbol)
        if feature_row.empty:
            emit_log("live_runtime", "feature_row_empty", console=False, symbol=symbol)
            return [], []
        traders = self._traders_by_symbol.get(symbol, [])
        if not traders:
            return [], []
        signal_candidates: list[Dict[str, Any]] = []
        close_candidates: list[Dict[str, Any]] = []
        for spec in traders:
            go_long = self._evaluate_rules(feature_row, spec.long_rules)
            go_short = self._evaluate_rules(feature_row, spec.short_rules)
            trader_positions = [
                p
                for p in open_positions_by_trader.get(spec.trader_id, [])
                if str(p.get("symbol") or "").upper() == symbol.upper()
            ]
            if not go_long and not go_short:
                for position in trader_positions:
                    close_candidates.append(
                        {
                            "spec": spec,
                            "position": dict(position),
                            "reason": "signal_inactive",
                        }
                    )
                emit_log(
                    "live_runtime",
                    "signal_not_triggered",
                    console=False,
                    trader_id=spec.trader_id,
                    symbol=symbol,
                )
                continue
            side = "buy" if go_long else "sell"
            signal_label = "SignalType.BUY" if side == "buy" else "SignalType.SELL"
            tf_label = self._tf_label()
            print(f"{self._dateprint()} - Recibido SIGNAL EVENT {signal_label} para {symbol} ({tf_label})")
            active_same_side = [p for p in trader_positions if self._position_direction(p) == side]
            active_opposite_side = [p for p in trader_positions if self._position_direction(p) and self._position_direction(p) != side]
            for position in active_opposite_side:
                close_candidates.append(
                    {
                        "spec": spec,
                        "position": dict(position),
                        "reason": "signal_side_changed",
                    }
                )
            if active_same_side:
                self._update_signal_book(
                    {
                        "trader_id": spec.trader_id,
                        "symbol": symbol.upper(),
                        "side": side,
                        "signal_label": signal_label,
                        "status": "already_open",
                        "selected": True,
                        "price": close_px,
                        "detected_at": datetime.now().isoformat(),
                        "reason": "position_already_open",
                        "portfolio_phase": self._current_portfolio_phase(),
                    }
                )
                continue
            signal_candidates.append(
                {
                    "trader_id": spec.trader_id,
                    "symbol": symbol.upper(),
                    "side": side,
                    "signal_label": signal_label,
                    "price": close_px,
                    "spec": spec,
                    "detected_at": datetime.now().isoformat(),
                }
            )
        return signal_candidates, close_candidates

    def _collect_manual_signal_snapshot(
        self,
        *,
        include_existing_positions: bool = True,
    ) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
        signal_candidates: list[Dict[str, Any]] = []
        close_candidates: list[Dict[str, Any]] = []
        open_positions_by_trader = self._open_positions_by_trader()
        for symbol, traders in self._traders_by_symbol.items():
            bars = self.data_provider.get_latest_closed_bars(symbol, timeframe=self.data_provider.timeframe, num_bars=self.bars_lookback)
            if bars.empty:
                emit_log("live_runtime", "feature_row_empty_manual_rebalance", console=False, symbol=symbol)
                continue
            ohlc = bars[["open", "high", "low", "close"]].copy()
            features = build_features(ohlc, dropna=True)
            if features.empty:
                emit_log("live_runtime", "feature_row_empty_manual_rebalance", console=False, symbol=symbol)
                continue
            feature_row = features.tail(1)
            close_px = float(pd.to_numeric(ohlc["close"], errors="coerce").dropna().iloc[-1]) if not ohlc.empty else 0.0
            for spec in traders:
                go_long = self._evaluate_rules(feature_row, spec.long_rules)
                go_short = self._evaluate_rules(feature_row, spec.short_rules)
                trader_positions = [
                    p
                    for p in open_positions_by_trader.get(spec.trader_id, [])
                    if str(p.get("symbol") or "").upper() == symbol.upper()
                ]
                if not go_long and not go_short:
                    for position in trader_positions:
                        close_candidates.append(
                            {
                                "spec": spec,
                                "position": dict(position),
                                "reason": "signal_inactive",
                            }
                        )
                    continue
                side = "buy" if go_long else "sell"
                signal_label = "SignalType.BUY" if side == "buy" else "SignalType.SELL"
                active_same_side = [p for p in trader_positions if self._position_direction(p) == side]
                active_opposite_side = [p for p in trader_positions if self._position_direction(p) and self._position_direction(p) != side]
                for position in active_opposite_side:
                    close_candidates.append(
                        {
                            "spec": spec,
                            "position": dict(position),
                            "reason": "signal_side_changed",
                        }
                    )
                signal_candidates.append(
                    {
                        "trader_id": spec.trader_id,
                        "symbol": symbol.upper(),
                        "side": side,
                        "signal_label": signal_label,
                        "price": close_px,
                        "spec": spec,
                        "detected_at": datetime.now().isoformat(),
                        "already_open": bool(active_same_side) and include_existing_positions,
                        "open_positions": [dict(p) for p in active_same_side],
                        "reason": "manual_ui_retrain",
                    }
                )
        return signal_candidates, close_candidates

    def force_rebalance_now(self, *, reason: str = "manual_ui_retrain") -> Dict[str, Any]:
        try:
            self.data_provider.check_for_new_data(force_emit_snapshot=True)
        except TypeError:
            try:
                self.data_provider.check_for_new_data()
            except Exception:
                pass
        signal_candidates, close_candidates = self._collect_manual_signal_snapshot(include_existing_positions=True)
        for candidate in close_candidates:
            self._attempt_close(
                spec=candidate["spec"],
                position=candidate["position"],
                source="signal",
                rationale=str(candidate.get("reason") or "signal_inactive"),
            )
        if not signal_candidates:
            self._last_portfolio_output = {
                "selected_tickers": [],
                "weights": {},
                "euros": {},
                "comparison": pd.DataFrame(),
                "figures": {},
                "status": "no_active_signals_for_manual_rebalance",
                "reason": "no_active_signals",
                "portfolio_phase": self._current_portfolio_phase(),
                "close_candidates_count": int(len(close_candidates)),
            }
            return dict(self._last_portfolio_output)
        self._process_signal_candidates(signal_candidates, force_rebalance=True, manual_reason=reason)
        return dict(self._last_portfolio_output or {})

    def poll_once(self) -> int:
        self._process_pending_retries()
        self.data_provider.check_for_new_data()
        processed = 0
        signal_candidates: list[Dict[str, Any]] = []
        close_candidates: list[Dict[str, Any]] = []
        open_positions_by_trader = self._open_positions_by_trader()
        while True:
            try:
                event = self._queue.get(block=False)
            except Empty:
                break
            if isinstance(event, DataEvent):
                processed += 1
                event_signals, event_closes = self._process_data_event(event, open_positions_by_trader)
                signal_candidates.extend(event_signals)
                close_candidates.extend(event_closes)
        for candidate in close_candidates:
            self._attempt_close(
                spec=candidate["spec"],
                position=candidate["position"],
                source="signal",
                rationale=str(candidate.get("reason") or "signal_inactive"),
            )
        if signal_candidates:
            self._process_signal_candidates(signal_candidates)
        return processed

    def run(self, *, max_cycles: int = 500, idle_sleep_sec: int = 60) -> None:
        emit_log(
            "live_runtime",
            "started",
            console=False,
            symbols=sorted(self._traders_by_symbol.keys()),
            timeframe=str(getattr(self.data_provider, "timeframe", self.timeframe)),
            max_cycles=max_cycles,
            idle_sleep_sec=idle_sleep_sec,
        )
        cycles = 0
        while cycles < max_cycles:
            cycles += 1
            processed = self.poll_once()
            if processed == 0:
                sleep(max(5, int(idle_sleep_sec)))
        emit_log("live_runtime", "finished", console=False, cycles=cycles)

