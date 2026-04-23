from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Mapping

from app.agents import DataAgent, DeveloperAgent, PortfolioManagerAgent, TraderAgent, ValidationAgent
from app.contracts import PromotedTraderSpec, TraderLifecycleState, TraderLiveMetrics
from app.core.structured_logging import emit_log


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CandidateBuildResult:
    asset: str
    dataset_id: str
    experiment_id: str
    trader_id: str
    n_long_rules: int
    n_short_rules: int


class SimulationRuntime:
    """
    Fase funcional preproduccion (sin MT5):
    - genera traders promovidos para un universo de activos
    - mantiene cola de promovidos en estado PROMOTED
    - publica metricas de scouting para que PortfolioManager decida activacion
    - activa solo un subconjunto a estado LIVE
    """

    def __init__(
        self,
        *,
        data_agent: DataAgent,
        developer_agent: DeveloperAgent,
        validation_agent: ValidationAgent,
        trader_agent: TraderAgent,
        portfolio_agent: PortfolioManagerAgent,
    ) -> None:
        self.data_agent = data_agent
        self.developer_agent = developer_agent
        self.validation_agent = validation_agent
        self.trader_agent = trader_agent
        self.portfolio_agent = portfolio_agent
        self._promoted_registry: Dict[str, PromotedTraderSpec] = {}

    def get_promoted_registry(self) -> Dict[str, PromotedTraderSpec]:
        return dict(self._promoted_registry)

    def _asset_strategy(self, asset: str) -> Dict[str, object]:
        seed = sum(ord(c) for c in asset.upper())
        chosen_family = ("decision_tree", "rulefit", "genetico")[seed % 3]
        family_params: Dict[str, Dict[str, object]]
        if chosen_family == "decision_tree":
            family_params = {
                "decision_tree": {"target_n_rules": 18 + (seed % 10), "progress_every": 0}
            }
        elif chosen_family == "rulefit":
            family_params = {
                "rulefit": {
                    "target_n_rules": 18 + ((seed // 3) % 10),
                    "n_estimators": 20 + (seed % 11),
                    "max_candidate_rules": 130 + (seed % 35),
                    "progress_every": 0,
                }
            }
        else:
            family_params = {
                "genetico": {
                    "target_n_rules": 18 + ((seed // 5) % 10),
                    "population_size": 26 + (seed % 15),
                    "n_generations": 6 + (seed % 5),
                    "progress_every": 0,
                }
            }

        split_config = {
            # Developer usa IS para construir reglas; Validation usa OOS para validarlas.
            "is_pct": round(0.55 + ((seed % 15) / 100.0), 2),  # 0.55..0.69
            "oos_pct": 0.0,  # se ajusta abajo para sumar 1.0
            "holdout_year": 2025,
            "lookback_years": 10,
        }
        split_config["oos_pct"] = round(1.0 - float(split_config["is_pct"]), 2)

        validation_profile = {
            "split_assumption": {"holdout_year": 2025},
            "monkey_is": {
                "n_monkeys": 100 + (seed % 50),
                "is_pass_pct": 85.0 + float(seed % 10),
                "min_coverage_is": 70 + (seed % 25),
                "n_jobs": 1,
            },
            "monkey_oos": {
                "n_monkeys": 90 + (seed % 40),
                "oos_pass_pct": 70.0 + float(seed % 10),
                "min_coverage_oos": 55 + (seed % 20),
                "n_jobs": 1,
            },
            "correlation_pruning": {
                "corr_threshold": round(0.45 + ((seed % 10) / 100.0), 2),
                "min_ops": 40 + (seed % 20),
                "diagnose": False,
            },
            "forward_validation": {
                "target_year": 2025,
                "min_ops": 25 + (seed % 15),
                "verbose": False,
            },
            "stability_selection": {
                "top_n_long": 10 + (seed % 8),
                "top_n_short": 10 + (seed % 8),
                "min_ops": 40 + (seed % 20),
                "verbose": False,
            },
        }

        return {
            "chosen_family": chosen_family,
            "family_params": family_params,
            "split_config": split_config,
            "validation_profile": validation_profile,
        }

    def _scouting_metrics(self, promoted: PromotedTraderSpec) -> TraderLiveMetrics:
        # Métricas sintéticas pre-activación: permiten ranking antes de LIVE real.
        seed = sum(ord(c) for c in promoted.trader_id)
        n_rules = len(promoted.long_rules) + len(promoted.short_rules)
        sharpe = round(0.10 + ((seed % 70) / 100.0), 4)  # 0.10..0.79
        drawdown = round(0.03 + ((seed % 22) / 200.0), 4)  # 0.03..0.14
        pnl = float(100 + (seed % 1200))
        corr_penalty = round(((seed % 30) / 100.0), 4)
        trades = 8 + (seed % 25)
        return TraderLiveMetrics(
            trader_id=promoted.trader_id,
            as_of=utc_now_iso(),
            pnl=pnl,
            sharpe_rolling=sharpe,
            drawdown_rolling=drawdown,
            trade_count=trades,
            extra_metrics={
                "corr_penalty": corr_penalty,
                "phase": "scouting_pre_live",
                "asset": promoted.asset,
                "n_rules": n_rules,
            },
        )

    def build_candidate_pool(
        self,
        *,
        asset_csv_by_asset: Mapping[str, str],
        timeframe: str = "D1",
    ) -> list[CandidateBuildResult]:
        out: list[CandidateBuildResult] = []
        emit_log(
            "simulation_runtime",
            "candidate_pool_build_started",
            n_assets=len(asset_csv_by_asset),
            assets=list(asset_csv_by_asset.keys()),
            timeframe=timeframe,
            model_selection_mode="single_family_per_asset",
        )

        for asset, csv_path in asset_csv_by_asset.items():
            strategy = self._asset_strategy(asset)
            chosen_family = str(strategy["chosen_family"])
            params = strategy["family_params"]
            split_config = strategy["split_config"]
            validation_profile = strategy["validation_profile"]
            emit_log(
                "simulation_runtime",
                "developer_strategy_selected",
                asset=asset,
                chosen_family=chosen_family,
                family_params=params,
                split_config=split_config,
            )
            emit_log(
                "simulation_runtime",
                "validation_strategy_selected",
                asset=asset,
                validation_profile=validation_profile,
            )
            dataset = self.data_agent.prepare_dataset(asset=asset, timeframe=timeframe, asset_csv_path=csv_path)
            dev = self.developer_agent.develop(
                dataset=dataset,
                families=(chosen_family,),
                family_params=params,
                split_config=split_config,
            )
            val = self.validation_agent.validate_and_promote(dev, validation_profile=validation_profile)
            promoted = val.promoted_spec
            self._promoted_registry[promoted.trader_id] = promoted

            scouting = self._scouting_metrics(promoted)
            self.trader_agent.publish_metrics(scouting, correlation_id=promoted.origin_experiment_id)

            row = CandidateBuildResult(
                asset=asset,
                dataset_id=dataset.dataset_id,
                experiment_id=dev.experiment_config.experiment_id,
                trader_id=promoted.trader_id,
                n_long_rules=len(promoted.long_rules),
                n_short_rules=len(promoted.short_rules),
            )
            out.append(row)
            emit_log(
                "simulation_runtime",
                "candidate_ready_promoted_queue",
                asset=asset,
                trader_id=promoted.trader_id,
                rules={"long": row.n_long_rules, "short": row.n_short_rules},
            )

        emit_log(
            "simulation_runtime",
            "candidate_pool_build_completed",
            promoted_count=len(out),
            trader_ids=[r.trader_id for r in out],
        )
        return out

    def activate_top_candidates(
        self,
        *,
        as_of: str | None = None,
        max_live_traders: int = 2,
        max_weight: float = 0.7,
        min_score: float = -0.25,
    ) -> list[str]:
        decision = self.portfolio_agent.rebalance(
            as_of=as_of or utc_now_iso(),
            max_weight=max_weight,
            min_score=min_score,
        )
        ranked = sorted(decision.weights.items(), key=lambda x: float(x[1]), reverse=True)
        selected_ids = [trader_id for trader_id, _ in ranked[:max_live_traders]]
        activated: list[str] = []

        for trader_id in selected_ids:
            state = self.data_agent.ctx.store.get_trader_state(trader_id)
            if state is None or state.state != TraderLifecycleState.PROMOTED:
                continue
            promoted = self._promoted_registry.get(trader_id)
            if promoted is None:
                continue
            self.trader_agent.activate(promoted)
            activated.append(trader_id)

        emit_log(
            "simulation_runtime",
            "activation_cycle_completed",
            requested_max_live=max_live_traders,
            portfolio_selected=decision.selected_traders,
            activated=activated,
        )
        return activated

