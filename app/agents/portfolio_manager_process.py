"""
PortfolioManagerProcess (modo unico GA + PSO).

Proceso semanal determinista de asignacion de capital. No es un agente
deliberativo: no aprende online, no genera senales primarias y no se ejecuta
de forma continua. Su flujo es un pipeline:

1. Recibe el universo de traders promovidos (`sync_universe`).
2. Recibe las senales activas en el rebalanceo semanal (`rebalance_active_signals`).
3. Construye una matriz de retornos historica (semanal, ventana configurable).
4. Llama al optimizador hibrido GA + PSO definido en
   `app.services.portfolio_optimizer` para seleccionar traders y asignar pesos.
5. Devuelve un PortfolioDecision trazable y persiste un snapshot de rebalanceo.

NO entrena modelos. NO carga checkpoints. NO contiene PPO, policy networks ni RL.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from app.contracts import AgentStatus, EventType, PortfolioDecision, PromotedTraderSpec
from app.core.structured_logging import emit_log
from app.services.portfolio_optimizer import (
    OptimizationResult,
    PortfolioOptimizerConfig,
    equal_weight_fitness,
    optimize_portfolio_ga_pso,
)
from app.services.portfolio_support import UniverseRegistry

from .base import AgentContext


class PortfolioManagerProcess:
    """
    Proceso semanal de optimizacion de cartera. Modo unico: GA + PSO.

    - `sync_universe(promoted_specs)`: refresca la lista canonica de traders promovidos.
    - `rebalance_active_signals(...)`: ejecuta la optimizacion para los traders con
      senal activa en el rebalanceo y devuelve la asignacion de capital.

    El identificador `agent_id` se mantiene como `"portfolio_manager"` por
    compatibilidad con la auditoria, los eventos y la tabla de control de
    acceso a ejecucion.
    """

    agent_id = "portfolio_manager"

    def __init__(
        self,
        ctx: AgentContext | None = None,
        *,
        optimizer_config: PortfolioOptimizerConfig | None = None,
        random_state: int = 42,
    ) -> None:
        self.ctx = ctx
        self.random_state = int(random_state)
        self.config = optimizer_config or PortfolioOptimizerConfig(random_seed=int(random_state))
        self.universe_registry = UniverseRegistry(self.ctx.store) if self.ctx is not None else None
        self._promoted_specs: Dict[str, PromotedTraderSpec] = {}
        self._last_optimization: OptimizationResult | None = None
        self._last_decision_payload: Dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Sincronizacion del universo
    # ------------------------------------------------------------------

    def sync_universe(self, promoted_specs: Mapping[str, PromotedTraderSpec]) -> None:
        self._promoted_specs = {str(k): v for k, v in promoted_specs.items()}
        if self.universe_registry is not None:
            self.universe_registry.sync_promoted_specs(self._promoted_specs)

    # ------------------------------------------------------------------
    # Carga y normalizacion de senales activas
    # ------------------------------------------------------------------

    @staticmethod
    def discover_active_systems(
        active_signals: Sequence[Mapping[str, Any]] | pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Normaliza la lista de senales activas a un DataFrame canonico."""
        if active_signals is None:
            return pd.DataFrame(columns=["trader_id", "symbol", "side", "signal_label", "price"])
        if isinstance(active_signals, pd.DataFrame):
            df = active_signals.copy()
        else:
            df = pd.DataFrame(list(active_signals))
        if df.empty:
            return pd.DataFrame(columns=["trader_id", "symbol", "side", "signal_label", "price"])
        if "asset" in df.columns and "symbol" not in df.columns:
            df = df.rename(columns={"asset": "symbol"})
        for col in ["trader_id", "symbol", "side"]:
            if col not in df.columns:
                raise ValueError(f"Falta columna requerida en senales activas: {col}")
        if "signal_label" not in df.columns:
            df["signal_label"] = df["side"].map(
                lambda x: "SignalType.BUY" if str(x).lower() == "buy" else "SignalType.SELL"
            )
        if "price" not in df.columns:
            df["price"] = np.nan
        df["trader_id"] = df["trader_id"].astype(str)
        df["symbol"] = df["symbol"].astype(str).str.upper()
        df["side"] = df["side"].astype(str).str.lower()
        df = df.drop_duplicates(subset=["trader_id", "symbol", "side"]).reset_index(drop=True)
        return df[["trader_id", "symbol", "side", "signal_label", "price"]]

    # ------------------------------------------------------------------
    # Carga de retornos
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_history_to_weekly_returns(
        history: pd.DataFrame | pd.Series,
        *,
        weekly_frequency: str,
        lookback_weeks: int,
        capital_base: float = 10000.0,
    ) -> pd.Series:
        """
        Convierte el historial de un trader (curva de equity / pnl) en una serie
        de retornos semanales reescalada y truncada a `lookback_weeks` semanas.
        """
        if isinstance(history, pd.Series):
            df = history.to_frame(name="value")
        else:
            df = history.copy()
        if df is None or df.empty:
            return pd.Series(dtype="float64")

        cols_lower = {str(c).lower(): c for c in df.columns}
        if "date" in cols_lower:
            date_col = cols_lower["date"]
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()].sort_index()

        value_col = None
        for candidate in (
            "weekly_return",
            "equity",
            "balance",
            "cum_pnl",
            "cumulative_pnl",
            "pnl",
            "value",
        ):
            if candidate in cols_lower:
                value_col = cols_lower[candidate]
                break
        if value_col is None:
            value_col = df.columns[0]

        series = pd.to_numeric(df[value_col], errors="coerce").dropna()
        if series.empty:
            return pd.Series(dtype="float64")

        col_lower = str(value_col).lower()
        if col_lower == "weekly_return":
            weekly = series.copy()
            weekly.index = pd.to_datetime(weekly.index, errors="coerce")
            weekly = weekly[~weekly.index.isna()].sort_index()
            weekly = weekly.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            if col_lower == "pnl":
                series = float(capital_base) + series
            series = series.replace([np.inf, -np.inf], np.nan).dropna()
            if series.empty:
                return pd.Series(dtype="float64")
            daily_returns = series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            weekly = (1.0 + daily_returns).resample(weekly_frequency).prod() - 1.0

        weekly = weekly.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if int(lookback_weeks) > 0 and len(weekly) > int(lookback_weeks):
            weekly = weekly.iloc[-int(lookback_weeks):]
        return weekly

    def load_system_returns(
        self,
        active_systems: Sequence[Mapping[str, Any]] | pd.DataFrame,
        *,
        history_loader: Callable[[str], pd.DataFrame | pd.Series | None] | None = None,
        historical_pnl_paths: Mapping[str, str | Path] | None = None,
        historical_series_by_system: Mapping[str, pd.DataFrame | pd.Series] | None = None,
    ) -> pd.DataFrame:
        """
        Carga los retornos semanales de cada trader activo y los alinea por fecha.
        Devuelve un DataFrame `(T, N)` con columnas = trader_id.
        """
        active_df = self.discover_active_systems(active_systems)
        if active_df.empty:
            return pd.DataFrame()

        series_map: Dict[str, pd.Series] = {}
        for trader_id in active_df["trader_id"].tolist():
            history_obj: Any = None
            if historical_series_by_system and trader_id in historical_series_by_system:
                history_obj = historical_series_by_system[trader_id]
            elif history_loader is not None:
                history_obj = history_loader(trader_id)
            elif historical_pnl_paths and trader_id in historical_pnl_paths:
                history_obj = pd.read_csv(historical_pnl_paths[trader_id])
            if history_obj is None:
                continue
            ret = self._coerce_history_to_weekly_returns(
                history_obj,
                weekly_frequency=self.config.weekly_frequency,
                lookback_weeks=self.config.lookback_weeks,
            )
            if ret.empty or float(ret.abs().sum()) == 0.0:
                continue
            series_map[trader_id] = ret.rename(trader_id)

        if not series_map:
            return pd.DataFrame()

        returns_df = pd.concat(series_map.values(), axis=1, join="outer").sort_index()
        returns_df = returns_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        nz_cols = [c for c in returns_df.columns if float(returns_df[c].abs().sum()) > 0.0]
        return returns_df[nz_cols]

    # ------------------------------------------------------------------
    # Rebalanceo principal (GA + PSO)
    # ------------------------------------------------------------------

    def rebalance_active_signals(
        self,
        *,
        active_signals: Sequence[Mapping[str, Any]] | pd.DataFrame,
        total_capital_eur: float,
        history_loader: Callable[[str], pd.DataFrame | pd.Series | None] | None = None,
        historical_pnl_paths: Mapping[str, str | Path] | None = None,
        historical_series_by_system: Mapping[str, pd.DataFrame | pd.Series] | None = None,
        frequency: str | None = None,  # ignorado: el agente impone weekly por config
        lookback: int | None = None,  # ignorado: usa config.lookback_weeks
    ) -> Dict[str, Any]:
        _ = (frequency, lookback)  # parametros aceptados por compatibilidad pero ignorados
        start_ts = time.perf_counter()
        active_df = self.discover_active_systems(active_signals)
        if active_df.empty:
            return self._empty_pm_output(active_df, status="no_active_signals", elapsed=0.0)

        returns_df = self.load_system_returns(
            active_df,
            history_loader=history_loader,
            historical_pnl_paths=historical_pnl_paths,
            historical_series_by_system=historical_series_by_system,
        )
        active_ids = active_df["trader_id"].astype(str).tolist()
        valid_ids = [c for c in returns_df.columns.tolist() if c in set(active_ids)] if not returns_df.empty else []
        excluded_ids = sorted(set(active_ids) - set(valid_ids))
        active_universe_size = len(active_ids)
        valid_universe_size = len(valid_ids)

        if valid_universe_size == 0:
            elapsed = time.perf_counter() - start_ts
            return self._empty_pm_output(
                active_df,
                status="no_history",
                elapsed=elapsed,
                excluded_traders=excluded_ids,
                active_universe_size=active_universe_size,
                valid_universe_size=0,
            )

        ordered_returns = returns_df[valid_ids]
        returns_matrix = ordered_returns.to_numpy(dtype=float, copy=False)
        result = optimize_portfolio_ga_pso(returns_matrix, config=self.config)
        self._last_optimization = result

        # Map columnas (indices) -> trader_id
        selected_traders = [str(valid_ids[i]) for i in result.selected_indices]
        weights_map: Dict[str, float] = {
            str(valid_ids[i]): float(w) for i, w in zip(result.selected_indices, result.weights)
        }
        cash_weight = float(result.cash_weight)
        euros_map: Dict[str, float] = {
            tid: float(w) * float(total_capital_eur) for tid, w in weights_map.items()
        }

        # Baseline equal weight (sobre todos los traders validos) para comparacion.
        eq_fit, eq_sharpe, eq_mdd, eq_corr = equal_weight_fitness(
            returns_matrix,
            lambda_dd=self.config.lambda_dd,
            lambda_corr=self.config.lambda_corr,
        )

        elapsed = time.perf_counter() - start_ts
        rebalance_ts = datetime.now(timezone.utc).isoformat()
        rebalance_id = f"rb_{uuid4().hex[:10]}"

        diagnostics: Dict[str, Any] = {
            "ga_top_subsets": [
                {
                    "indices": [str(valid_ids[i]) for i in sub.indices],
                    "fitness": float(sub.fitness),
                    "sharpe": float(sub.sharpe),
                    "mdd": float(sub.mdd),
                    "corr_media": float(sub.corr_media),
                }
                for sub in result.ga_top_subsets
            ],
            "pso_iterations": int(result.pso_iterations),
            "status": str(result.status),
        }

        baselines = {
            "equal_weight_all_valid": {
                "fitness": float(eq_fit),
                "sharpe_neto": float(eq_sharpe),
                "mdd": float(eq_mdd),
                "corr_media": float(eq_corr),
            }
        }

        metadata: Dict[str, Any] = {
            "optimizer_mode": "ga_pso",
            "ga_config": {
                "population_size": int(self.config.ga_population_size),
                "generations": int(self.config.ga_generations),
                "tournament_size": int(self.config.ga_tournament_size),
                "crossover_rate": float(self.config.ga_crossover_rate),
                "mutation_rate": float(self.config.ga_mutation_rate),
                "elitism": int(self.config.ga_elitism),
                "early_stopping_generations": int(self.config.ga_early_stopping_generations),
                "top_k_subsets_for_pso": int(self.config.top_k_subsets_for_pso),
            },
            "pso_config": {
                "swarm_size": int(self.config.pso_swarm_size),
                "iterations": int(self.config.pso_iterations),
                "inertia_start": float(self.config.pso_inertia_start),
                "inertia_end": float(self.config.pso_inertia_end),
                "cognitive_coef": float(self.config.pso_cognitive_coef),
                "social_coef": float(self.config.pso_social_coef),
                "early_stopping_iterations": int(self.config.pso_early_stopping_iterations),
            },
            "lambda_dd": float(self.config.lambda_dd),
            "lambda_corr": float(self.config.lambda_corr),
            "lookback_weeks": int(self.config.lookback_weeks),
            "weekly_frequency": str(self.config.weekly_frequency),
            "min_selected_traders": int(self.config.min_selected_traders),
            "max_selected_traders": int(self.config.max_selected_traders),
            "max_weight_per_trader": float(self.config.max_weight_per_trader),
            "min_live_weight": float(self.config.min_live_weight),
            "max_cash_weight": float(self.config.max_cash_weight),
            "active_universe_size": int(active_universe_size),
            "valid_universe_size": int(valid_universe_size),
            "selected_universe_size": int(len(selected_traders)),
            "excluded_traders": list(excluded_ids),
            "execution_time_seconds": float(elapsed),
            "baselines": baselines,
            "ppo_removed": True,
        }

        decision = PortfolioDecision(
            decision_id=rebalance_id,
            as_of=rebalance_ts,
            selected_traders=list(selected_traders),
            weights={k: float(v) for k, v in weights_map.items()},
            rationale="ga_pso optimization with fitness Sharpe - lambda_dd*MDD - lambda_corr*CorrMedia",
            optimizer_mode="ga_pso",
            target_cash_weight=float(cash_weight),
            active_universe_size=int(active_universe_size),
            valid_universe_size=int(valid_universe_size),
            selected_universe_size=int(len(selected_traders)),
            fitness=float(result.fitness),
            sharpe_neto=float(result.sharpe_neto),
            mdd=float(result.mdd),
            corr_media=float(result.corr_media),
            metadata=metadata,
        )
        decision_payload = decision.to_dict()
        self._last_decision_payload = decision_payload

        # Curva historica de la cartera seleccionada y comparativa equal-weight.
        portfolio_curve = self._build_portfolio_curves(
            returns_df=ordered_returns,
            selected_traders=selected_traders,
            weights_map=weights_map,
        )

        # Persistencia.
        if self.ctx is not None:
            try:
                self.ctx.store.set_agent_status(
                    self.agent_id, AgentStatus.RUNNING, "ga_pso portfolio optimization"
                )
                self.ctx.store.append_event(
                    event_id=f"evt_{uuid4().hex[:10]}",
                    event_type=EventType.PORTFOLIO_DECISION,
                    producer=self.agent_id,
                    payload=decision_payload,
                    correlation_id=rebalance_id,
                )
                self.ctx.store.upsert_portfolio_rebalance_snapshot(
                    rebalance_id=rebalance_id,
                    rebalance_date=rebalance_ts,
                    active_traders=active_ids,
                    selected_traders=list(selected_traders),
                    target_weights={k: float(v) for k, v in weights_map.items()},
                    target_cash_weight=float(cash_weight),
                    diagnostics=diagnostics,
                    forward_metrics={},
                    metadata={
                        **metadata,
                        "fitness": float(result.fitness),
                        "sharpe_neto": float(result.sharpe_neto),
                        "mdd": float(result.mdd),
                        "corr_media": float(result.corr_media),
                        "metrics_json": {
                            "fitness": float(result.fitness),
                            "sharpe_neto": float(result.sharpe_neto),
                            "mdd": float(result.mdd),
                            "corr_media": float(result.corr_media),
                            "selected_traders": list(selected_traders),
                            "weights": {k: float(v) for k, v in weights_map.items()},
                            "cash": float(cash_weight),
                            "baseline_metrics": baselines,
                            "ppo_removed": True,
                        },
                    },
                )
                emit_log(
                    self.agent_id,
                    "portfolio_decision_ga_pso",
                    console=False,
                    decision=decision_payload,
                )
                self.ctx.store.set_agent_status(
                    self.agent_id, AgentStatus.IDLE, "ga_pso portfolio optimization done"
                )
            except Exception as exc:
                emit_log(self.agent_id, "portfolio_persist_error", console=False, error=str(exc))

        return {
            "active_systems": active_df,
            "selected_tickers": list(selected_traders),
            "weights": {k: float(v) for k, v in weights_map.items()},
            "euros": {k: float(v) for k, v in euros_map.items()},
            "risk_contribution": {k: float(v) for k, v in weights_map.items()},
            "target_cash_weight": float(cash_weight),
            "decision": decision_payload,
            "status": str(result.status if result.status else "ok"),
            "optimizer_mode": "ga_pso",
            "fitness": float(result.fitness),
            "sharpe_neto": float(result.sharpe_neto),
            "mdd": float(result.mdd),
            "corr_media": float(result.corr_media),
            "active_universe_size": int(active_universe_size),
            "valid_universe_size": int(valid_universe_size),
            "selected_universe_size": int(len(selected_traders)),
            "execution_time_seconds": float(elapsed),
            "diagnostics": diagnostics,
            "baselines": baselines,
            "returns_active": ordered_returns,
            "returns_selected": ordered_returns[selected_traders] if selected_traders else pd.DataFrame(index=ordered_returns.index),
            "portfolio_curve": portfolio_curve,
            "selection_df": self._build_selection_df(active_df, weights_map, euros_map),
            "comparison": pd.DataFrame(),
            "figures": {},
        }

    # Compatibilidad con scripts antiguos (no realiza ningun trabajo).
    def rebalance(self, *, as_of: str, **_: Any) -> PortfolioDecision:
        decision = PortfolioDecision(
            decision_id=f"pm_{uuid4().hex[:10]}",
            as_of=as_of,
            selected_traders=[],
            weights={},
            rationale="legacy entry-point disabled; use rebalance_active_signals",
            optimizer_mode="ga_pso",
        )
        if self.ctx is not None:
            self.ctx.store.append_event(
                event_id=f"evt_{uuid4().hex[:10]}",
                event_type=EventType.PORTFOLIO_DECISION,
                producer=self.agent_id,
                payload=decision.to_dict(),
                correlation_id=decision.decision_id,
            )
        return decision

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------

    def _empty_pm_output(
        self,
        active_df: pd.DataFrame,
        *,
        status: str,
        elapsed: float,
        excluded_traders: list[str] | None = None,
        active_universe_size: int = 0,
        valid_universe_size: int = 0,
    ) -> Dict[str, Any]:
        return {
            "active_systems": active_df,
            "selected_tickers": [],
            "weights": {},
            "euros": {},
            "risk_contribution": {},
            "target_cash_weight": 1.0,
            "decision": PortfolioDecision(
                decision_id=f"rb_{uuid4().hex[:10]}",
                as_of=datetime.now(timezone.utc).isoformat(),
                selected_traders=[],
                weights={},
                rationale="ga_pso optimization not run (no active or valid universe)",
                optimizer_mode="ga_pso",
                target_cash_weight=1.0,
                active_universe_size=int(active_universe_size or len(active_df.index)),
                valid_universe_size=int(valid_universe_size),
                selected_universe_size=0,
                metadata={
                    "optimizer_mode": "ga_pso",
                    "excluded_traders": list(excluded_traders or []),
                    "execution_time_seconds": float(elapsed),
                    "ppo_removed": True,
                },
            ).to_dict(),
            "status": str(status),
            "optimizer_mode": "ga_pso",
            "fitness": 0.0,
            "sharpe_neto": 0.0,
            "mdd": 0.0,
            "corr_media": 0.0,
            "active_universe_size": int(active_universe_size or len(active_df.index)),
            "valid_universe_size": int(valid_universe_size),
            "selected_universe_size": 0,
            "execution_time_seconds": float(elapsed),
            "diagnostics": {"status": str(status)},
            "baselines": {},
            "returns_active": pd.DataFrame(),
            "returns_selected": pd.DataFrame(),
            "portfolio_curve": pd.DataFrame(),
            "selection_df": pd.DataFrame(),
            "comparison": pd.DataFrame(),
            "figures": {},
        }

    @staticmethod
    def _build_selection_df(
        active_df: pd.DataFrame,
        weights_map: Mapping[str, float],
        euros_map: Mapping[str, float],
    ) -> pd.DataFrame:
        if active_df is None or active_df.empty:
            return pd.DataFrame()
        out = active_df.copy()
        out["selected"] = out["trader_id"].isin(set(weights_map.keys()))
        out["weight"] = out["trader_id"].map(lambda tid: float(weights_map.get(str(tid), 0.0)))
        out["euros"] = out["trader_id"].map(lambda tid: float(euros_map.get(str(tid), 0.0)))
        out["risk_contribution"] = out["weight"]
        return out

    @staticmethod
    def _build_portfolio_curves(
        returns_df: pd.DataFrame,
        *,
        selected_traders: Sequence[str],
        weights_map: Mapping[str, float],
    ) -> pd.DataFrame:
        if returns_df is None or returns_df.empty:
            return pd.DataFrame()
        if not selected_traders:
            equal_curve = (1.0 + returns_df.mean(axis=1)).cumprod() - 1.0
            return pd.DataFrame({"equal_weight": equal_curve})
        sel = [c for c in selected_traders if c in returns_df.columns]
        if not sel:
            equal_curve = (1.0 + returns_df.mean(axis=1)).cumprod() - 1.0
            return pd.DataFrame({"equal_weight": equal_curve})
        weights_series = pd.Series(
            {tid: float(weights_map.get(str(tid), 0.0)) for tid in sel},
            dtype=float,
        )
        weights_total = float(weights_series.sum())
        if weights_total <= 0.0:
            weights_series = pd.Series(1.0 / float(len(sel)), index=sel)
        else:
            weights_series = weights_series / weights_total
        portfolio_returns = returns_df[sel].mul(weights_series, axis=1).sum(axis=1)
        equal_returns = returns_df[sel].mean(axis=1)
        portfolio_curve = (1.0 + portfolio_returns).cumprod() - 1.0
        equal_curve = (1.0 + equal_returns).cumprod() - 1.0
        return pd.DataFrame(
            {
                "ga_pso": portfolio_curve,
                "equal_weight": equal_curve,
            }
        )

    # ------------------------------------------------------------------
    # Pasarelas a broker (solo lectura). Mantenidas por compatibilidad.
    # ------------------------------------------------------------------

    def get_broker_positions(self) -> list[dict]:
        if self.ctx is None or self.ctx.execution_router is None:
            return []
        return self.ctx.execution_router.get_open_positions(actor=self.agent_id)

    def get_market_snapshot(self, symbol: str) -> dict:
        if self.ctx is None or self.ctx.execution_router is None:
            return {}
        return self.ctx.execution_router.get_market_snapshot(actor=self.agent_id, symbol=symbol)
