from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from app.core.structured_logging import emit_log
from app.execution.models import OrderIntent, OrderSide


class MT5Connector:
    """
    Adaptador mínimo para MT5 basado en el framework externo.
    La conexión es opt-in; si falla, el router puede quedarse en modo paper.
    """

    def __init__(self, *, env_path: str = ".env") -> None:
        self.env_path = Path(env_path)
        self._mt5 = None
        self._connected = False
        self._config: Dict[str, Any] = {}
        self._marketwatch_reported: set[str] = set()
        self._marketwatch_warned: set[str] = set()

    @staticmethod
    def _dateprint() -> str:
        return datetime.now().strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]

    @staticmethod
    def _retcode_name(mt5_module: Any, retcode: int) -> str:
        for attr in dir(mt5_module):
            if not attr.startswith("TRADE_RETCODE_"):
                continue
            try:
                if int(getattr(mt5_module, attr)) == int(retcode):
                    return attr
            except Exception:
                continue
        return "TRADE_RETCODE_UNKNOWN"

    @staticmethod
    def _safe_comment(comment: str) -> str:
        raw = str(comment or "").strip()
        if not raw:
            return "tfm-mkt"
        safe = []
        for ch in raw:
            if ch.isalnum() or ch in {"-", "_"}:
                safe.append(ch)
            else:
                safe.append("_")
        out = "".join(safe)
        if len(out) > 31:
            out = out[:31]
        return out or "tfm-mkt"

    @staticmethod
    def _magic_from_trader_id(trader_id: str) -> int:
        txt = str(trader_id or "")
        digits = "".join(ch for ch in txt if ch.isdigit())
        if digits:
            try:
                return int(digits[-9:])
            except Exception:
                pass
        return abs(hash(txt)) % 900_000_000 + 100_000_000

    def _build_live_comment(self, trader_id: str, requested_comment: str) -> str:
        requested = str(requested_comment or "").strip()
        if requested.upper().startswith("MANUAL"):
            return self._safe_comment(requested)
        magic = self._magic_from_trader_id(trader_id)
        return self._safe_comment(f"{magic}-MKT")

    def _select_filling_type(self, symbol: str) -> int | None:
        if self._mt5 is None:
            return None
        order_fillings: list[int] = []
        for name in ("ORDER_FILLING_FOK", "ORDER_FILLING_IOC", "ORDER_FILLING_RETURN"):
            if hasattr(self._mt5, name):
                try:
                    order_fillings.append(int(getattr(self._mt5, name)))
                except Exception:
                    continue
        if not order_fillings:
            return None

        try:
            info = self._mt5.symbol_info(symbol)
        except Exception:
            info = None
        if info is None:
            return order_fillings[0]

        filling_mode = getattr(info, "filling_mode", None)
        if filling_mode is not None:
            try:
                fm = int(filling_mode)
                if fm in order_fillings:
                    return fm
            except Exception:
                pass
        return order_fillings[0]

    def _filling_candidates(self, symbol: str) -> list[int | None]:
        if self._mt5 is None:
            return [None]
        candidates: list[int | None] = []
        preferred = self._select_filling_type(symbol)
        if preferred is not None:
            candidates.append(preferred)
        for name in ("ORDER_FILLING_FOK", "ORDER_FILLING_IOC", "ORDER_FILLING_RETURN"):
            if hasattr(self._mt5, name):
                try:
                    val = int(getattr(self._mt5, name))
                    if val not in candidates:
                        candidates.append(val)
                except Exception:
                    continue
        candidates.append(None)
        return candidates

    def _normalize_volume(self, symbol: str, requested_volume: float) -> float:
        if self._mt5 is None:
            return float(max(0.0, requested_volume))
        try:
            info = self._mt5.symbol_info(symbol)
        except Exception:
            info = None
        if info is None:
            return float(max(0.0, requested_volume))
        volume = float(max(0.0, requested_volume))
        volume_min = float(getattr(info, "volume_min", 0.01) or 0.01)
        volume_max = float(getattr(info, "volume_max", volume) or volume)
        volume_step = float(getattr(info, "volume_step", volume_min) or volume_min)
        volume = max(volume, volume_min)
        if volume_step > 0:
            steps = round(volume / volume_step)
            volume = steps * volume_step
        volume = max(volume, volume_min)
        if volume_max > 0:
            volume = min(volume, volume_max)
        return float(f"{volume:.8f}")

    def _is_success_retcode(self, retcode: int) -> bool:
        if self._mt5 is None:
            return False
        accepted_names = ["TRADE_RETCODE_DONE", "TRADE_RETCODE_NO_CHANGES", "TRADE_RETCODE_DONE_PARTIAL"]
        accepted_values = set()
        for name in accepted_names:
            if hasattr(self._mt5, name):
                try:
                    accepted_values.add(int(getattr(self._mt5, name)))
                except Exception:
                    continue
        return int(retcode) in accepted_values

    def _check_algo_trading_enabled(self) -> None:
        if not self._connected or self._mt5 is None:
            return
        try:
            terminal_info = self._mt5.terminal_info()
        except Exception:
            terminal_info = None
        if terminal_info is None:
            raise RuntimeError("No se pudo obtener terminal_info() de MT5.")
        if not bool(getattr(terminal_info, "trade_allowed", False)):
            raise RuntimeError("El trading algorítmico está desactivado en MT5.")

    def _build_request(
        self,
        *,
        intent: OrderIntent,
        order_type: int,
        normalized_volume: float,
        comment: str,
        include_price: bool,
        filling_type: int | None,
        deviation: int,
    ) -> Dict[str, Any]:
        request: Dict[str, Any] = {
            "action": self._mt5.TRADE_ACTION_DEAL,
            "symbol": intent.symbol,
            "volume": float(normalized_volume),
            "type": order_type,
            "magic": self._magic_from_trader_id(intent.trader_id),
            "comment": comment,
            "deviation": int(deviation),
        }
        if include_price:
            tick = self._mt5.symbol_info_tick(intent.symbol)
            if tick is not None:
                request["price"] = float(tick.ask if intent.side == OrderSide.BUY else tick.bid)
        if intent.sl is not None and float(intent.sl) > 0.0:
            request["sl"] = float(intent.sl)
        if intent.tp is not None and float(intent.tp) > 0.0:
            request["tp"] = float(intent.tp)
        if filling_type is not None:
            request["type_filling"] = int(filling_type)
        return request

    def _extract_result_payload(self, result: Any, request: Dict[str, Any], check_payload: Dict[str, Any] | None) -> Dict[str, Any]:
        payload = result._asdict()
        payload["order_check"] = check_payload
        payload["request"] = request
        payload["last_error"] = str(self._mt5.last_error())
        retcode = int(payload.get("retcode", -1))
        payload["retcode_name"] = self._retcode_name(self._mt5, retcode)
        return payload

    @staticmethod
    def _safe_last_error(mt5_module: Any) -> str:
        try:
            return str(mt5_module.last_error())
        except Exception:
            return "unavailable"

    @staticmethod
    def _looks_like_server_path_mismatch(path: str | None, server: str | None) -> bool:
        path_txt = str(path or "").strip().lower()
        server_txt = str(server or "").strip().lower()
        if not path_txt or not server_txt:
            return False
        return ("darwinex" in path_txt and "metaquotes" in server_txt) or (
            "metaquotes" in path_txt and "darwinex" in server_txt
        )

    def _candidate_terminal_paths(self) -> list[str]:
        candidates: list[str] = []

        def add_candidate(raw_path: str | Path | None) -> None:
            txt = str(raw_path or "").strip()
            if not txt:
                return
            path = Path(txt)
            if not path.exists() or not path.is_file():
                return
            resolved = str(path.resolve())
            if resolved not in candidates:
                candidates.append(resolved)

        add_candidate(self._config.get("path"))
        for raw_path in (
            r"C:\Program Files\MetaTrader 5\terminal64.exe",
            r"C:\Program Files\MetaTrader 5\terminal.exe",
            r"C:\Program Files\MetaTrader 5\terminal32.exe",
            r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
            r"C:\Program Files (x86)\MetaTrader 5\terminal.exe",
            r"C:\Program Files (x86)\MetaTrader 5\terminal32.exe",
            r"C:\Program Files\Darwinex MetaTrader 5\terminal64.exe",
            r"C:\Program Files\Darwinex MetaTrader 5\terminal.exe",
            r"C:\Program Files (x86)\Darwinex MetaTrader 5\terminal64.exe",
            r"C:\Program Files (x86)\Darwinex MetaTrader 5\terminal.exe",
        ):
            add_candidate(raw_path)

        roaming_root = Path.home() / "AppData" / "Roaming" / "MetaQuotes" / "Terminal"
        if roaming_root.exists():
            for pattern in ("terminal64.exe", "terminal.exe"):
                for match in roaming_root.rglob(pattern):
                    add_candidate(match)

        return candidates

    def _build_initialize_kwargs(self, *, path: str | None, include_credentials: bool, timeout_ms: int) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "timeout": int(timeout_ms),
            "portable": self._config["portable"],
        }
        if path:
            kwargs["path"] = path
        if include_credentials:
            if int(self._config.get("login") or 0) > 0:
                kwargs["login"] = self._config["login"]
            if self._config.get("password"):
                kwargs["password"] = self._config["password"]
            if self._config.get("server"):
                kwargs["server"] = self._config["server"]
        return kwargs

    def _launch_terminal_if_possible(self, path: str | None) -> bool:
        exe_path = Path(str(path or "").strip())
        if not exe_path.exists() or not exe_path.is_file():
            return False
        try:
            subprocess.Popen([str(exe_path)], cwd=str(exe_path.parent))
            time.sleep(4.0)
            return True
        except Exception:
            return False

    def _attempt_initialize(
        self,
        mt5_module: Any,
        *,
        path: str | None,
        include_credentials: bool,
        launch_before: bool,
        timeout_ms: int,
    ) -> Dict[str, Any]:
        try:
            mt5_module.shutdown()
        except Exception:
            pass

        launched = False
        if launch_before and path:
            launched = self._launch_terminal_if_possible(path)

        kwargs = self._build_initialize_kwargs(path=path, include_credentials=include_credentials, timeout_ms=timeout_ms)
        attempt: Dict[str, Any] = {
            "path": path,
            "include_credentials": include_credentials,
            "launch_before": launch_before,
            "launched": launched,
        }
        try:
            ok = bool(mt5_module.initialize(**kwargs))
        except Exception as exc:
            attempt["ok"] = False
            attempt["error"] = str(exc)
            attempt["last_error"] = self._safe_last_error(mt5_module)
            return attempt

        attempt["ok"] = ok
        attempt["last_error"] = self._safe_last_error(mt5_module)
        return attempt

    def _wait_for_deals(self, order_ticket: int | str | None) -> list[Dict[str, Any]]:
        if self._mt5 is None or not order_ticket:
            return []
        deals = self._mt5.history_deals_get(position=order_ticket)
        if not deals:
            tries = 0
            while tries < 100:
                time.sleep(0.05)
                deals = self._mt5.history_deals_get(position=order_ticket)
                if deals:
                    break
                tries += 1
        if not deals:
            return []
        out: list[Dict[str, Any]] = []
        for d in deals:
            try:
                out.append(d._asdict())
            except Exception:
                continue
        return out

    def connect(self, *, quick: bool = False) -> bool:
        try:
            from dotenv import find_dotenv, load_dotenv
            import MetaTrader5 as mt5
        except Exception as exc:
            emit_log("mt5_connector", "connect_import_failed", env_path=str(self.env_path), error=str(exc))
            return False

        load_dotenv(dotenv_path=find_dotenv(str(self.env_path)) or self.env_path)
        self._config = {
            "path": os.getenv("MT5_PATH"),
            "login": int(os.getenv("MT5_LOGIN", "0")),
            "password": os.getenv("MT5_PASSWORD"),
            "server": os.getenv("MT5_SERVER"),
            "timeout": int(os.getenv("MT5_TIMEOUT", "60000")),
            "portable": str(os.getenv("MT5_PORTABLE", "False")).strip().lower() == "true",
        }

        missing_fields = []
        if int(self._config["login"]) <= 0:
            missing_fields.append("MT5_LOGIN")
        if not str(self._config.get("password") or "").strip():
            missing_fields.append("MT5_PASSWORD")
        if not str(self._config.get("server") or "").strip():
            missing_fields.append("MT5_SERVER")
        if missing_fields:
            emit_log(
                "mt5_connector",
                "connect_config_warning",
                console=False,
                env_path=str(self.env_path),
                missing_fields=missing_fields,
            )

        candidate_paths = self._candidate_terminal_paths()
        mismatch_hint = self._looks_like_server_path_mismatch(self._config.get("path"), self._config.get("server"))
        if mismatch_hint:
            emit_log(
                "mt5_connector",
                "connect_config_warning",
                console=False,
                env_path=str(self.env_path),
                configured_path=self._config.get("path"),
                server=self._config.get("server"),
                reason="configured_path_looks_mismatched_with_server",
            )

        plans: list[tuple[str | None, bool, bool]] = []
        has_credentials = not missing_fields
        quick_timeout_ms = max(1000, min(int(self._config["timeout"]), 5000))
        full_timeout_ms = int(self._config["timeout"])
        timeout_ms = quick_timeout_ms if quick else full_timeout_ms
        candidate_paths_to_try = candidate_paths[:1] if quick and candidate_paths else candidate_paths
        if has_credentials:
            for path in candidate_paths_to_try:
                plans.append((path, True, False))
                if not quick:
                    plans.append((path, True, True))
            plans.append((None, True, False))
        for path in candidate_paths_to_try:
            plans.append((path, False, False))
        plans.append((None, False, False))

        attempts: list[Dict[str, Any]] = []
        seen_plans: set[tuple[str | None, bool, bool]] = set()
        selected_attempt: Dict[str, Any] | None = None
        for path, include_credentials, launch_before in plans:
            plan_key = (path, include_credentials, launch_before)
            if plan_key in seen_plans:
                continue
            seen_plans.add(plan_key)
            attempt = self._attempt_initialize(
                mt5,
                path=path,
                include_credentials=include_credentials,
                launch_before=launch_before,
                timeout_ms=timeout_ms,
            )
            attempts.append(attempt)
            if bool(attempt.get("ok")):
                selected_attempt = attempt
                break

        if selected_attempt is None:
            emit_log(
                "mt5_connector",
                "connect_failed",
                env_path=str(self.env_path),
                configured_path=self._config.get("path"),
                server=self._config.get("server"),
                login=self._config.get("login"),
                quick=quick,
                candidate_paths=candidate_paths,
                server_path_mismatch_hint=mismatch_hint,
                attempts=attempts,
            )
            return False

        self._config["resolved_path"] = selected_attempt.get("path")
        self._config["connected_without_credentials"] = not bool(selected_attempt.get("include_credentials"))
        self._mt5 = mt5
        self._connected = True
        print("MT5 conectado exitosamente")
        self._print_live_account_guard()
        self._print_account_info_human()
        try:
            self._check_algo_trading_enabled()
        except Exception:
            self.shutdown()
            raise
        emit_log(
            "mt5_connector",
            "connected",
            console=False,
            env_path=str(self.env_path),
            configured_path=self._config.get("path"),
            resolved_path=self._config.get("resolved_path"),
            server=self._config.get("server"),
            login=self._config.get("login"),
            quick=quick,
            connected_without_credentials=self._config.get("connected_without_credentials", False),
            selected_attempt=selected_attempt,
            mt5_version=str(mt5.version()),
        )
        account_info = self.account_info()
        emit_log("mt5_connector", "account_info", console=False, info=account_info)
        expected_login = int(self._config.get("login") or 0)
        actual_login = int(account_info.get("login") or 0)
        if expected_login > 0 and actual_login > 0 and expected_login != actual_login:
            emit_log(
                "mt5_connector",
                "connected_login_mismatch",
                console=False,
                expected_login=expected_login,
                actual_login=actual_login,
                resolved_path=self._config.get("resolved_path"),
            )
        return True

    def _print_live_account_guard(self) -> None:
        if not self._connected or self._mt5 is None:
            return
        info = self._mt5.account_info()
        if info is None:
            return
        trade_mode = int(getattr(info, "trade_mode", -1))
        if trade_mode == int(self._mt5.ACCOUNT_TRADE_MODE_REAL):
            confirm = str(os.getenv("MT5_REAL_ACCOUNT_CONFIRM", "y")).strip().lower()
            print(f"ALERTA! Cuenta de tipo REAL detectada. Capital en riesgo. ¿Desea continuar? (y/n): {confirm}")
            if confirm != "y":
                self.shutdown()
                raise RuntimeError("Ejecución detenida por MT5_REAL_ACCOUNT_CONFIRM != y")

    def _print_account_info_human(self) -> None:
        info = self.account_info()
        if not info.get("connected"):
            return
        print("+----------- Información de la cuenta -----------")
        print(f"| - ID de la cuenta: {info.get('login')}")
        print(f"| - Nombre del trader: {info.get('name')}")
        print(f"| - Broker: {info.get('company')}")
        print(f"| - Servidor: {info.get('server')}")
        print(f"| - Tipo de cuenta: {info.get('trade_mode')} (0=Demo, 1=Contest, 2=Real)")
        print(f"| - Modo de margen: {info.get('margin_mode')} (0=Netting, 1=Exchange, 2=Hedge)")
        print(f"| - Apalancamiento: {info.get('leverage')}")
        print(f"| - Divisa de la cuenta: {info.get('currency')}")
        print(f"| - Balance de la cuenta: {info.get('balance')}")
        print(f"| - Patrimonio: {info.get('equity')}")
        print("+------------------------------------------------")

    def ensure_symbols_in_marketwatch(self, symbols: list[str]) -> None:
        if not self._connected or self._mt5 is None:
            return
        for symbol in symbols:
            info = self._mt5.symbol_info(symbol)
            if info is None:
                if self._mt5.symbol_select(symbol, True):
                    if symbol not in self._marketwatch_reported:
                        print(f"{self._dateprint()} - INFO: Símbolo {symbol} se ha añadido con éxito al MarketWatch.")
                        self._marketwatch_reported.add(symbol)
                else:
                    if symbol not in self._marketwatch_warned:
                        print(f"{self._dateprint()} - AVISO: no se ha podido añadir el símbolo {symbol} al MarketWatch.")
                        self._marketwatch_warned.add(symbol)
                continue
            if bool(getattr(info, "visible", False)):
                if symbol not in self._marketwatch_reported:
                    print(f"{self._dateprint()} - INFO: El símbolo {symbol} ya estaba en el MarketWatch.")
                    self._marketwatch_reported.add(symbol)
            else:
                if self._mt5.symbol_select(symbol, True):
                    if symbol not in self._marketwatch_reported:
                        print(f"{self._dateprint()} - INFO: Símbolo {symbol} se ha añadido con éxito al MarketWatch.")
                        self._marketwatch_reported.add(symbol)
                else:
                    if symbol not in self._marketwatch_warned:
                        print(f"{self._dateprint()} - AVISO: no se ha podido añadir el símbolo {symbol} al MarketWatch.")
                        self._marketwatch_warned.add(symbol)

    def shutdown(self) -> None:
        if self._mt5 is not None:
            try:
                self._mt5.shutdown()
            except Exception:
                pass
        self._connected = False
        emit_log("mt5_connector", "shutdown", console=False)

    @property
    def connected(self) -> bool:
        return self._connected

    def account_info(self) -> Dict[str, Any]:
        if not self._connected or self._mt5 is None:
            return {"connected": False}
        info = self._mt5.account_info()
        if info is None:
            return {"connected": True, "account_info": None}
        data = info._asdict()
        return {
            "connected": True,
            "login": data.get("login"),
            "name": data.get("name"),
            "server": data.get("server"),
            "company": data.get("company"),
            "trade_mode": data.get("trade_mode"),
            "margin_mode": data.get("margin_mode"),
            "leverage": data.get("leverage"),
            "currency": data.get("currency"),
            "balance": data.get("balance"),
            "equity": data.get("equity"),
        }

    def send_market_order(self, intent: OrderIntent) -> Dict[str, Any]:
        if not self._connected or self._mt5 is None:
            return {"ok": False, "reason": "mt5_not_connected"}
        self.ensure_symbols_in_marketwatch([intent.symbol])
        order_type = self._mt5.ORDER_TYPE_BUY if intent.side == OrderSide.BUY else self._mt5.ORDER_TYPE_SELL
        comment = self._build_live_comment(intent.trader_id, intent.comment)
        volume = self._normalize_volume(intent.symbol, float(intent.volume))

        # Intento principal: request alineada con mt5-framework/pyeventbt
        # (sin price, deviation=0, filling FOK si está disponible).
        requests_to_try: list[Dict[str, Any]] = []
        seen_requests: set[str] = set()
        for filling_type in self._filling_candidates(intent.symbol):
            for include_price, deviation in ((False, 0), (True, int(os.getenv("MT5_DEVIATION", "20")))):
                req = self._build_request(
                    intent=intent,
                    order_type=order_type,
                    normalized_volume=volume,
                    comment=comment,
                    include_price=include_price,
                    filling_type=filling_type,
                    deviation=deviation,
                )
                sig = str(sorted(req.items()))
                if sig not in seen_requests:
                    seen_requests.add(sig)
                    requests_to_try.append(req)

        final_payload: Dict[str, Any] | None = None
        final_reason = "order_send_none"
        for request in requests_to_try:
            check_payload: Dict[str, Any] | None = None
            try:
                check_result = self._mt5.order_check(request)
                if check_result is not None:
                    check_payload = check_result._asdict()
            except Exception:
                check_payload = None

            result = self._mt5.order_send(request)
            if result is None:
                final_payload = {
                    "retcode": None,
                    "retcode_name": None,
                    "comment": "order_send returned None",
                    "retcode_external": None,
                    "last_error": str(self._mt5.last_error()),
                    "order_check": check_payload,
                    "request": request,
                }
                final_reason = "order_send_none"
                continue

            payload = self._extract_result_payload(result, request, check_payload)
            retcode = int(payload.get("retcode", -1))
            ok = self._is_success_retcode(retcode)
            if ok:
                payload["deals"] = self._wait_for_deals(payload.get("order") or payload.get("deal"))
                return {"ok": True, "reason": "mt5_order_sent", "payload": payload}
            final_payload = payload
            final_reason = f"mt5_rejected_{payload.get('retcode_name')}({retcode})"

        if final_payload is None:
            final_payload = {"last_error": str(self._mt5.last_error())}

        reason = "mt5_order_sent"
        if final_reason != "mt5_order_sent":
            reason = final_reason
        return {"ok": False, "reason": reason, "payload": final_payload}

    def close_position(self, *, position: Dict[str, Any], trader_id: str, comment: str = "") -> Dict[str, Any]:
        if not self._connected or self._mt5 is None:
            return {"ok": False, "reason": "mt5_not_connected"}

        symbol = str(position.get("symbol") or "").upper()
        if not symbol:
            return {"ok": False, "reason": "position_symbol_missing"}
        self.ensure_symbols_in_marketwatch([symbol])

        pos_type = int(position.get("type", -1))
        if pos_type == int(self._mt5.ORDER_TYPE_BUY):
            close_type = self._mt5.ORDER_TYPE_SELL
        elif pos_type == int(self._mt5.ORDER_TYPE_SELL):
            close_type = self._mt5.ORDER_TYPE_BUY
        else:
            return {"ok": False, "reason": "position_type_invalid", "payload": {"position": position}}

        volume = self._normalize_volume(symbol, float(position.get("volume") or 0.0))
        if volume <= 0:
            return {"ok": False, "reason": "position_volume_invalid", "payload": {"position": position}}

        close_comment = self._safe_comment(comment or f"{self._magic_from_trader_id(trader_id)}-CLS")
        requests_to_try: list[Dict[str, Any]] = []
        seen_requests: set[str] = set()
        for filling_type in self._filling_candidates(symbol):
            req = {
                "action": self._mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(volume),
                "type": close_type,
                "position": int(position.get("ticket") or position.get("identifier") or 0),
                "magic": self._magic_from_trader_id(trader_id),
                "comment": close_comment,
                "deviation": int(os.getenv("MT5_DEVIATION", "20")),
            }
            if filling_type is not None:
                req["type_filling"] = int(filling_type)
            tick = self._mt5.symbol_info_tick(symbol)
            if tick is not None:
                req["price"] = float(tick.bid if close_type == self._mt5.ORDER_TYPE_SELL else tick.ask)
            sig = str(sorted(req.items()))
            if sig not in seen_requests:
                seen_requests.add(sig)
                requests_to_try.append(req)

        final_payload: Dict[str, Any] | None = None
        final_reason = "order_send_none"
        for request in requests_to_try:
            check_payload: Dict[str, Any] | None = None
            try:
                check_result = self._mt5.order_check(request)
                if check_result is not None:
                    check_payload = check_result._asdict()
            except Exception:
                check_payload = None

            result = self._mt5.order_send(request)
            if result is None:
                final_payload = {
                    "retcode": None,
                    "retcode_name": None,
                    "comment": "order_send returned None",
                    "retcode_external": None,
                    "last_error": str(self._mt5.last_error()),
                    "order_check": check_payload,
                    "request": request,
                }
                final_reason = "order_send_none"
                continue

            payload = self._extract_result_payload(result, request, check_payload)
            retcode = int(payload.get("retcode", -1))
            if self._is_success_retcode(retcode):
                payload["deals"] = self._wait_for_deals(payload.get("order") or payload.get("deal"))
                return {"ok": True, "reason": "mt5_close_sent", "payload": payload}
            final_payload = payload
            final_reason = f"mt5_close_rejected_{payload.get('retcode_name')}({retcode})"

        if final_payload is None:
            final_payload = {"last_error": str(self._mt5.last_error())}
        return {"ok": False, "reason": final_reason, "payload": final_payload}

    def get_open_positions(self) -> list[Dict[str, Any]]:
        if not self._connected or self._mt5 is None:
            return []
        pos = self._mt5.positions_get()
        if pos is None:
            return []
        out: list[Dict[str, Any]] = []
        for p in pos:
            out.append(p._asdict())
        return out

