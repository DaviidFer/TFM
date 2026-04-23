from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from app.contracts import AgentStatus, EventType, PortfolioDecision, PromotedTraderSpec
from app.core.structured_logging import emit_log
from app.services.portfolio_rl import (
    PPOInferenceService,
    PPOPortfolioConfig,
    PPOTrainer,
    PortfolioArtifactsManager,
    PortfolioDataset,
    PortfolioDatasetBuilder,
    UniverseRegistry,
)

from .base import AgentContext


@dataclass(frozen=True)
class GeneticSelectionResult:
    selected_tickers: list[str]
    condition_evolution: list[float]
    best_cov_matrix: pd.DataFrame
    best_corr_matrix: pd.DataFrame
    n_selected: int
    best_fitness: float


class PortfolioManagerAgent:
    """
    Agente de portfolio management para señales activas.

    Filosofía:
    - Selección SOLO sobre sistemas con señal activa en el rebalanceo actual.
    - El genético usa SOLO el número de condición como fitness.
    - HRP se usa exclusivamente para weighting tras la selección.
    """

    agent_id = "portfolio_manager"

    def __init__(self, ctx: AgentContext | None = None, *, random_state: int = 42) -> None:
        self.ctx = ctx
        self.random_state = int(random_state)
        self._rng = np.random.default_rng(self.random_state)
        self.config = PPOPortfolioConfig(
            portfolio_manager_mode=str(os.getenv("PORTFOLIO_MANAGER_MODE", "ppo")).strip().lower() or "ppo"
        )
        self.artifacts = PortfolioArtifactsManager(self.config)
        self.dataset_builder = PortfolioDatasetBuilder(self.config, self.ctx.store if self.ctx is not None else None)
        self.trainer = PPOTrainer(self.config, self.artifacts)
        self.inference_service = PPOInferenceService(self.config, self.artifacts)
        self.universe_registry = UniverseRegistry(self.ctx.store) if self.ctx is not None else None
        self._promoted_specs: Dict[str, PromotedTraderSpec] = {}
        self._latest_dataset: PortfolioDataset | None = None
        self._latest_model_info: Dict[str, Any] | None = None
        self._latest_training_run: Dict[str, Any] | None = None

    @staticmethod
    def discover_active_systems(active_signals: Sequence[Mapping[str, Any]] | pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Devuelve el subconjunto de sistemas activos en el rebalanceo actual.

        Espera registros con columnas/keys como:
        - trader_id
        - symbol / asset
        - side
        - signal_label (opcional)
        - price (opcional)
        """
        if active_signals is None:
            return pd.DataFrame(columns=["trader_id", "symbol", "side", "signal_label", "price"])
        if isinstance(active_signals, pd.DataFrame):
            df = active_signals.copy()
        else:
            df = pd.DataFrame(list(active_signals))
        if df.empty:
            return pd.DataFrame(columns=["trader_id", "symbol", "side", "signal_label", "price"])

        rename_map = {}
        if "asset" in df.columns and "symbol" not in df.columns:
            rename_map["asset"] = "symbol"
        if rename_map:
            df = df.rename(columns=rename_map)
        for col in ["trader_id", "symbol", "side"]:
            if col not in df.columns:
                raise ValueError(f"Falta columna requerida en señales activas: {col}")
        if "signal_label" not in df.columns:
            df["signal_label"] = df["side"].map(lambda x: "SignalType.BUY" if str(x).lower() == "buy" else "SignalType.SELL")
        if "price" not in df.columns:
            df["price"] = np.nan
        df["trader_id"] = df["trader_id"].astype(str)
        df["symbol"] = df["symbol"].astype(str).str.upper()
        df["side"] = df["side"].astype(str).str.lower()
        df = df.drop_duplicates(subset=["trader_id", "symbol", "side"]).reset_index(drop=True)
        return df[["trader_id", "symbol", "side", "signal_label", "price"]]

    @staticmethod
    def _coerce_history_to_returns(
        history: pd.DataFrame | pd.Series,
        *,
        frequency: str = "daily",
        lookback: int = 252,
        capital_base: float = 10000.0,
    ) -> pd.Series:
        if isinstance(history, pd.Series):
            df = history.to_frame(name="value")
        else:
            df = history.copy()
        if df.empty:
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
        for candidate in ("equity", "balance", "cum_pnl", "cumulative_pnl", "pnl", "value"):
            if candidate in cols_lower:
                value_col = cols_lower[candidate]
                break
        if value_col is None:
            value_col = df.columns[0]

        series = pd.to_numeric(df[value_col], errors="coerce").dropna()
        if series.empty:
            return pd.Series(dtype="float64")

        if str(value_col).lower() == "pnl":
            series = float(capital_base) + series
        series = series.replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            return pd.Series(dtype="float64")

        returns = series.pct_change().replace([np.inf, -np.inf], np.nan)
        returns = returns.fillna(0.0)
        if frequency.lower().startswith("w"):
            returns = (1.0 + returns).resample("W-FRI").prod() - 1.0
        else:
            returns = returns.resample("B").last().ffill().pct_change().fillna(0.0)

        if int(lookback) > 0 and len(returns) > int(lookback):
            returns = returns.iloc[-int(lookback):]
        returns.name = str(value_col)
        return returns

    def load_system_returns(
        self,
        active_systems: Sequence[Mapping[str, Any]] | pd.DataFrame,
        *,
        historical_pnl_paths: Mapping[str, str | Path] | None = None,
        historical_series_by_system: Mapping[str, pd.DataFrame | pd.Series] | None = None,
        history_loader: Callable[[str], pd.DataFrame | pd.Series | None] | None = None,
        frequency: str = "daily",
        lookback: int = 252,
        capital_base: float = 10000.0,
    ) -> pd.DataFrame:
        """
        Carga y alinea la matriz de retornos de los sistemas activos.
        """
        active_df = self.discover_active_systems(active_systems)
        if active_df.empty:
            return pd.DataFrame()

        series_map: Dict[str, pd.Series] = {}
        for trader_id in active_df["trader_id"].tolist():
            history_obj = None
            if historical_series_by_system and trader_id in historical_series_by_system:
                history_obj = historical_series_by_system[trader_id]
            elif history_loader is not None:
                history_obj = history_loader(trader_id)
            elif historical_pnl_paths and trader_id in historical_pnl_paths:
                history_obj = pd.read_csv(historical_pnl_paths[trader_id])

            if history_obj is None:
                continue
            ret = self._coerce_history_to_returns(
                history_obj,
                frequency=frequency,
                lookback=lookback,
                capital_base=capital_base,
            )
            if ret.empty or float(ret.abs().sum()) == 0.0:
                continue
            series_map[trader_id] = ret.rename(trader_id)

        if not series_map:
            return pd.DataFrame()

        returns_df = pd.concat(series_map.values(), axis=1, join="outer").sort_index()
        returns_df = returns_df.replace([np.inf, -np.inf], np.nan)
        returns_df = returns_df.ffill(limit=3).fillna(0.0)
        nz_cols = [c for c in returns_df.columns if float(returns_df[c].abs().sum()) > 0.0]
        returns_df = returns_df[nz_cols]
        return returns_df

    @staticmethod
    def condition_number_fitness(
        returns_df: pd.DataFrame,
        *,
        epsilon: float = 1e-8,
        matrix_type: str = "cov",
    ) -> tuple[float, pd.DataFrame, pd.DataFrame]:
        """
        Fitness basado SOLO en la cercanía del número de condición a 1.
        """
        if returns_df.empty or returns_df.shape[1] == 0:
            raise ValueError("No hay retornos para calcular fitness.")
        work = returns_df.copy()
        work = work.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        cov = work.cov()
        cov_values = cov.to_numpy(dtype=float)
        cov_values = cov_values + (float(epsilon) * np.eye(cov_values.shape[0]))

        corr = work.corr().fillna(0.0)
        np.fill_diagonal(corr.values, 1.0)
        corr_values = corr.to_numpy(dtype=float)
        corr_values = corr_values + (float(epsilon) * np.eye(corr_values.shape[0]))

        matrix = cov_values if str(matrix_type).lower() == "cov" else corr_values
        try:
            eigenvalues = np.linalg.eigvalsh(matrix)
            eigenvalues = np.clip(eigenvalues, float(epsilon), None)
            condition_number = float(np.max(eigenvalues) / np.min(eigenvalues))
        except np.linalg.LinAlgError:
            condition_number = float("inf")
        fitness = -abs(condition_number - 1.0)
        cov_df = pd.DataFrame(cov_values, index=returns_df.columns, columns=returns_df.columns)
        corr_df = pd.DataFrame(corr_values, index=returns_df.columns, columns=returns_df.columns)
        return fitness, cov_df, corr_df

    def _repair_individual(self, individual: np.ndarray, min_selected: int) -> np.ndarray:
        ind = individual.astype(int).copy()
        if int(ind.sum()) == 0:
            pick = self._rng.integers(0, len(ind))
            ind[pick] = 1
        if int(ind.sum()) < int(min_selected):
            zeros = np.where(ind == 0)[0].tolist()
            self._rng.shuffle(zeros)
            need = int(min_selected) - int(ind.sum())
            for idx in zeros[:need]:
                ind[idx] = 1
        return ind

    def _generate_population(self, population_size: int, total_assets: int, min_selected: int) -> np.ndarray:
        population: list[np.ndarray] = []
        for _ in range(int(population_size)):
            selection_size = int(self._rng.integers(low=min_selected, high=max(min_selected + 1, total_assets + 1)))
            idx = self._rng.choice(total_assets, size=selection_size, replace=False)
            individual = np.zeros(total_assets, dtype=int)
            individual[idx] = 1
            population.append(self._repair_individual(individual, min_selected))
        return np.asarray(population, dtype=int)

    def _tournament_selection(self, population: np.ndarray, fitnesses: np.ndarray, k: int = 3) -> np.ndarray:
        k_eff = min(int(k), len(population))
        idx = self._rng.choice(len(population), size=k_eff, replace=False)
        winner = idx[int(np.argmax(fitnesses[idx]))]
        return population[winner].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray, min_selected: int) -> np.ndarray:
        if len(parent1) <= 1:
            return self._repair_individual(parent1.copy(), min_selected)
        point = int(self._rng.integers(1, len(parent1)))
        child = np.concatenate([parent1[:point], parent2[point:]])
        return self._repair_individual(child, min_selected)

    def _mutate(self, individual: np.ndarray, mutation_rate: float, min_selected: int) -> np.ndarray:
        mutated = individual.copy()
        for i in range(len(mutated)):
            if float(self._rng.random()) < float(mutation_rate):
                mutated[i] = 1 - mutated[i]
        return self._repair_individual(mutated, min_selected)

    def genetic_select_universe(
        self,
        returns_df: pd.DataFrame,
        *,
        population_size: int | None = None,
        generations: int = 60,
        mutation_rate: float = 0.15,
        tournament_k: int = 3,
        epsilon: float = 1e-8,
    ) -> GeneticSelectionResult:
        """
        Selección genética con cardinalidad variable y fitness SOLO por número de condición.
        """
        if returns_df.empty or returns_df.shape[1] == 0:
            return GeneticSelectionResult([], [], pd.DataFrame(), pd.DataFrame(), 0, float("-inf"))
        columns = list(returns_df.columns)
        total_assets = len(columns)
        min_selected = min(10, total_assets) if total_assets >= 10 else total_assets
        population_size = int(population_size or max(30, total_assets * 3))
        population = self._generate_population(population_size, total_assets, min_selected)

        best_individual = population[0].copy()
        best_fitness = float("-inf")
        best_cov = pd.DataFrame()
        best_corr = pd.DataFrame()
        condition_evolution: list[float] = []

        for _ in range(int(generations)):
            fitnesses = np.full(len(population), fill_value=float("-inf"))
            generation_best_cond = float("inf")
            for i, individual in enumerate(population):
                selected_cols = [columns[j] for j in range(total_assets) if int(individual[j]) == 1]
                if len(selected_cols) == 0:
                    continue
                subset = returns_df[selected_cols]
                fit, cov_df, corr_df = self.condition_number_fitness(subset, epsilon=epsilon, matrix_type="cov")
                fitnesses[i] = fit
                cond_val = abs(float(fit)) + 1.0
                generation_best_cond = min(generation_best_cond, cond_val)
                if float(fit) > float(best_fitness):
                    best_fitness = float(fit)
                    best_individual = individual.copy()
                    best_cov = cov_df
                    best_corr = corr_df
            condition_evolution.append(float(generation_best_cond))

            next_population = [best_individual.copy()]
            while len(next_population) < population_size:
                p1 = self._tournament_selection(population, fitnesses, k=tournament_k)
                p2 = self._tournament_selection(population, fitnesses, k=tournament_k)
                child = self._crossover(p1, p2, min_selected)
                child = self._mutate(child, mutation_rate, min_selected)
                next_population.append(child)
            population = np.asarray(next_population, dtype=int)

        selected = [columns[j] for j in range(total_assets) if int(best_individual[j]) == 1]
        if selected and best_cov.empty:
            _, best_cov, best_corr = self.condition_number_fitness(returns_df[selected], epsilon=epsilon, matrix_type="cov")
        return GeneticSelectionResult(
            selected_tickers=selected,
            condition_evolution=condition_evolution,
            best_cov_matrix=best_cov,
            best_corr_matrix=best_corr,
            n_selected=len(selected),
            best_fitness=float(best_fitness),
        )

    @staticmethod
    def _correl_distance(corr: pd.DataFrame) -> pd.DataFrame:
        return np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))

    @staticmethod
    def _quasi_diag(link: np.ndarray) -> list[int]:
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix.loc[i] = link[j, 0]
            df1 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df1]).sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    @staticmethod
    def _cluster_variance(cov: pd.DataFrame, cluster_items: list[str]) -> float:
        subcov = cov.loc[cluster_items, cluster_items]
        inv_diag = 1.0 / np.clip(np.diag(subcov.values), 1e-12, None)
        weights = inv_diag / inv_diag.sum()
        return float(np.dot(weights, np.dot(subcov.values, weights)))

    def allocate_weights_hrp(
        self,
        returns_df: pd.DataFrame,
        *,
        total_capital_eur: float = 100000.0,
    ) -> Dict[str, Any]:
        """
        Asigna pesos con HRP sobre un subconjunto ya seleccionado.
        """
        if returns_df.empty or returns_df.shape[1] == 0:
            return {
                "weights": {},
                "euros": {},
                "risk_contribution": {},
                "cov": pd.DataFrame(),
                "corr": pd.DataFrame(),
                "linkage": None,
                "ordered_assets": [],
            }
        if returns_df.shape[1] == 1:
            asset = str(returns_df.columns[0])
            return {
                "weights": {asset: 1.0},
                "euros": {asset: float(total_capital_eur)},
                "risk_contribution": {asset: 1.0},
                "cov": returns_df.cov().fillna(0.0),
                "corr": pd.DataFrame([[1.0]], index=[asset], columns=[asset]),
                "linkage": None,
                "ordered_assets": [asset],
            }
        cov = returns_df.cov().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        corr = returns_df.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        np.fill_diagonal(corr.values, 1.0)
        dist = self._correl_distance(corr)
        link = linkage(squareform(dist.values, checks=False), method="single")
        sorted_idx = self._quasi_diag(link)
        ordered_assets = corr.index[sorted_idx].tolist()

        weights = pd.Series(1.0, index=ordered_assets, dtype=float)
        clusters = [ordered_assets]
        while clusters:
            cluster = clusters.pop(0)
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            c1, c2 = cluster[:split], cluster[split:]
            var1 = self._cluster_variance(cov, c1)
            var2 = self._cluster_variance(cov, c2)
            alpha = 1.0 - (var1 / max(var1 + var2, 1e-12))
            weights[c1] *= alpha
            weights[c2] *= (1.0 - alpha)
            clusters.extend([c1, c2])
        weights = weights / weights.sum()
        euros = (weights * float(total_capital_eur)).to_dict()

        cov_np = cov.loc[weights.index, weights.index].values
        w_np = weights.values
        portfolio_vol = float(np.sqrt(max(np.dot(w_np, np.dot(cov_np, w_np)), 1e-12)))
        marginal = np.dot(cov_np, w_np) / portfolio_vol
        rc = (w_np * marginal) / portfolio_vol
        risk_contribution = {weights.index[i]: float(rc[i]) for i in range(len(weights))}
        return {
            "weights": {k: float(v) for k, v in weights.to_dict().items()},
            "euros": {k: float(v) for k, v in euros.items()},
            "risk_contribution": risk_contribution,
            "cov": cov,
            "corr": corr,
            "linkage": link,
            "ordered_assets": ordered_assets,
        }

    @staticmethod
    def allocate_weights_equal(
        returns_df: pd.DataFrame,
        *,
        total_capital_eur: float = 100000.0,
    ) -> Dict[str, Any]:
        if returns_df.empty or returns_df.shape[1] == 0:
            return {"weights": {}, "euros": {}, "risk_contribution": {}}
        n = returns_df.shape[1]
        weight = 1.0 / float(n)
        weights = {col: weight for col in returns_df.columns}
        euros = {col: weight * float(total_capital_eur) for col in returns_df.columns}
        return {
            "weights": weights,
            "euros": euros,
            "risk_contribution": {col: weight for col in returns_df.columns},
        }

    def compare_selected_vs_all_active(
        self,
        *,
        active_returns: pd.DataFrame,
        selected_returns: pd.DataFrame,
        hrp_all: Mapping[str, Any],
        hrp_selected: Mapping[str, Any],
        equal_selected: Mapping[str, Any],
        epsilon: float = 1e-8,
    ) -> pd.DataFrame:
        """
        Compara HRP sobre todo el universo activo vs subconjunto seleccionado.
        """
        rows: list[Dict[str, Any]] = []
        for name, ret_df, weights in [
            ("HRP all active", active_returns, hrp_all.get("weights", {})),
            ("HRP selected", selected_returns, hrp_selected.get("weights", {})),
            ("Equal selected", selected_returns, equal_selected.get("weights", {})),
        ]:
            if ret_df.empty or ret_df.shape[1] == 0:
                continue
            _, _, corr_df = self.condition_number_fitness(ret_df, epsilon=epsilon, matrix_type="cov")
            upper = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(bool))
            avg_corr = float(np.nanmean(upper.to_numpy()))
            cond_num = float(abs(self.condition_number_fitness(ret_df, epsilon=epsilon, matrix_type="cov")[0]) + 1.0)
            cov = ret_df.cov().fillna(0.0)
            w = pd.Series(weights, dtype=float).reindex(ret_df.columns).fillna(0.0)
            if float(w.sum()) <= 0:
                w[:] = 1.0 / len(w)
            w = w / w.sum()
            port_vol = float(np.sqrt(max(np.dot(w.values, np.dot(cov.values, w.values)), 1e-12)))
            diag_vol = np.sqrt(np.clip(np.diag(cov.values), 1e-12, None))
            div_ratio = float(np.dot(w.values, diag_vol) / max(port_vol, 1e-12))
            rows.append(
                {
                    "portfolio": name,
                    "n_assets": int(ret_df.shape[1]),
                    "condition_number": cond_num,
                    "avg_correlation": avg_corr,
                    "estimated_volatility": port_vol,
                    "diversification_ratio": div_ratio,
                }
            )
        return pd.DataFrame(rows)

    def plot_portfolio_manager_dashboard(
        self,
        *,
        condition_evolution: Sequence[float],
        corr_selected: pd.DataFrame,
        linkage_matrix: np.ndarray | None,
        weights_eur: Mapping[str, float],
        risk_contribution: Mapping[str, float],
        comparison_df: pd.DataFrame,
        rolling_curves: pd.DataFrame | None = None,
        weighted_curve: pd.Series | None = None,
    ) -> Dict[str, Figure]:
        """
        Genera gráficos del portfolio manager.
        """
        figs: Dict[str, Figure] = {}

        fig1, ax1 = plt.subplots(figsize=(8, 3))
        ax1.plot(list(range(1, len(condition_evolution) + 1)), list(condition_evolution), color="tab:blue")
        ax1.set_title("Evolución número de condición")
        ax1.set_xlabel("Generación")
        ax1.set_ylabel("Condition number")
        ax1.grid(True, alpha=0.25)
        figs["condition_evolution"] = fig1

        fig2, ax2 = plt.subplots(figsize=(6, 5))
        if not corr_selected.empty:
            im = ax2.imshow(corr_selected.values, vmin=-1.0, vmax=1.0, cmap="coolwarm")
            ax2.set_xticks(range(len(corr_selected.columns)))
            ax2.set_xticklabels(corr_selected.columns, rotation=90, fontsize=8)
            ax2.set_yticks(range(len(corr_selected.index)))
            ax2.set_yticklabels(corr_selected.index, fontsize=8)
            ax2.set_title("Heatmap correlación seleccionados")
            fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        else:
            ax2.text(0.5, 0.5, "Sin datos", ha="center", va="center")
            ax2.set_axis_off()
        figs["correlation_heatmap"] = fig2

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        if linkage_matrix is not None and corr_selected.shape[1] >= 2:
            dendrogram(linkage_matrix, labels=list(corr_selected.columns), ax=ax3, leaf_rotation=90)
            ax3.set_title("Dendrograma HRP")
        else:
            ax3.text(0.5, 0.5, "Dendrograma no disponible", ha="center", va="center")
            ax3.set_axis_off()
        figs["dendrogram"] = fig3

        fig4, ax4 = plt.subplots(figsize=(6, 3))
        if weights_eur:
            keys = list(weights_eur.keys())
            vals = [float(weights_eur[k]) for k in keys]
            plot_labels = self._pretty_plot_label_map(keys)
            ax4.bar(plot_labels, vals, color="tab:green")
            ax4.set_title("Asignación en euros")
            ax4.tick_params(axis="x", rotation=90)
            ax4.tick_params(axis="x", labelsize=8)
        else:
            ax4.text(0.5, 0.5, "Sin pesos", ha="center", va="center")
            ax4.set_axis_off()
        figs["weights_eur"] = fig4

        fig5, ax5 = plt.subplots(figsize=(6, 3))
        if risk_contribution:
            keys = list(risk_contribution.keys())
            vals = [float(risk_contribution[k]) for k in keys]
            plot_labels = self._pretty_plot_label_map(keys)
            ax5.bar(plot_labels, vals, color="tab:orange")
            ax5.set_title("Contribución al riesgo")
            ax5.tick_params(axis="x", rotation=90)
            ax5.tick_params(axis="x", labelsize=8)
        else:
            ax5.text(0.5, 0.5, "Sin contribución al riesgo", ha="center", va="center")
            ax5.set_axis_off()
        figs["risk_contribution"] = fig5

        fig6, ax6 = plt.subplots(figsize=(8, 4))
        if not comparison_df.empty:
            comp = comparison_df.set_index("portfolio")[["condition_number", "avg_correlation", "estimated_volatility", "diversification_ratio"]]
            comp.plot(kind="bar", ax=ax6)
            ax6.set_title("Comparativa selected vs all active")
            ax6.tick_params(axis="x", rotation=30)
            ax6.grid(True, alpha=0.2)
        else:
            ax6.text(0.5, 0.5, "Sin comparativa", ha="center", va="center")
            ax6.set_axis_off()
        figs["comparison"] = fig6

        fig7, ax7 = plt.subplots(figsize=(6, 3))
        plotted = False
        if rolling_curves is not None and not rolling_curves.empty:
            for col in rolling_curves.columns:
                ax7.plot(
                    rolling_curves.index,
                    rolling_curves[col],
                    alpha=0.4,
                    label=self._pretty_plot_trader_name(col),
                )
                plotted = True
        if weighted_curve is not None and not weighted_curve.empty:
            ax7.plot(weighted_curve.index, weighted_curve.values, color="black", linewidth=2.0, label="Portfolio PM")
            plotted = True
        if plotted:
            ax7.set_title("P/L rolling sistemas y cartera PM")
            ax7.grid(True, alpha=0.2)
            ax7.legend(fontsize=8, ncol=2)
        else:
            ax7.text(0.5, 0.5, "Sin curvas rolling", ha="center", va="center")
            ax7.set_axis_off()
        figs["rolling_curves"] = fig7
        return figs

    def sync_universe(
        self,
        promoted_specs: Mapping[str, PromotedTraderSpec],
    ) -> None:
        self._promoted_specs = {str(k): v for k, v in promoted_specs.items()}
        if self.universe_registry is not None:
            self.universe_registry.sync_promoted_specs(self._promoted_specs)

    @staticmethod
    def _month_token(as_of: str | None = None) -> str:
        ref = pd.Timestamp(as_of or datetime.now(timezone.utc).isoformat())
        return f"{ref.year:04d}-{ref.month:02d}"

    def _latest_model_from_store(self) -> Dict[str, Any] | None:
        if self.ctx is None:
            return self._latest_model_info
        try:
            latest = self.ctx.store.get_latest_portfolio_model_info()
            if latest:
                self._latest_model_info = latest
            return latest
        except Exception:
            return self._latest_model_info

    def _build_master_dataset(
        self,
        *,
        history_loader: Callable[[str], pd.DataFrame | pd.Series | None] | None,
    ) -> PortfolioDataset:
        if not self._promoted_specs:
            raise ValueError("No hay universo maestro de traders promovidos para Portfolio PPO.")
        dataset = self.dataset_builder.build_dataset(
            promoted_specs=self._promoted_specs,
            history_loader=history_loader,
        )
        self._latest_dataset = dataset
        return dataset

    def _dataset_refresh_metadata(self) -> Dict[str, Any]:
        if self._latest_dataset is None:
            return {}
        return dict(self._latest_dataset.trade_metadata.get("dataset_refresh") or {})

    def _enforce_min_open_positions(
        self,
        *,
        ppo_out: Dict[str, Any],
        active_df: pd.DataFrame,
        total_capital_eur: float,
    ) -> Dict[str, Any]:
        active_ids = [str(x) for x in active_df["trader_id"].astype(str).tolist()]
        if not active_ids:
            return ppo_out
        min_required = min(len(active_ids), max(1, int(self.config.min_open_positions)))
        current_selected = [tid for tid in list(ppo_out.get("selected_tickers") or []) if tid in set(active_ids)]
        if len(current_selected) >= min_required:
            return ppo_out

        weights_map = {str(k): float(v) for k, v in dict(ppo_out.get("weights") or {}).items()}
        rank_order = {tid: idx for idx, tid in enumerate(active_ids)}
        chosen = sorted(active_ids, key=lambda tid: (-float(weights_map.get(tid, 0.0)), rank_order[tid]))[:min_required]
        per_trader_weight = min(float(self.config.max_weight_per_trader), 1.0 / float(min_required))
        adjusted_weights = {tid: float(per_trader_weight) for tid in chosen}
        total_exposure = float(sum(adjusted_weights.values()))
        adjusted_cash = max(0.0, 1.0 - total_exposure)

        diagnostics = dict(ppo_out.get("diagnostics") or {})
        diagnostics["min_open_positions_enforced"] = True
        diagnostics["min_open_positions_target"] = int(min_required)
        diagnostics["min_open_positions_chosen"] = list(chosen)

        return {
            **ppo_out,
            "selected_tickers": list(chosen),
            "weights": adjusted_weights,
            "euros": {trader_id: float(weight * total_capital_eur) for trader_id, weight in adjusted_weights.items()},
            "target_cash_weight": float(adjusted_cash),
            "selected_universe_size": int(len(chosen)),
            "diagnostics": diagnostics,
        }

    def _persist_training_result(
        self,
        *,
        run_type: str,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        model_info = {
            "model_version": str(result["model_version"]),
            "mode": "ppo",
            "checkpoint_path": str(result["checkpoint_path"]),
            "universe_size": int(self._latest_dataset.n_traders if self._latest_dataset is not None else 0),
            "trained_at": pd.Timestamp.utcnow().isoformat() if run_type == "initial_train" else str((self._latest_model_info or {}).get("trained_at") or ""),
            "fine_tuned_at": pd.Timestamp.utcnow().isoformat() if run_type != "initial_train" else "",
            "config": self.config.to_dict(),
            "metrics": {
                "train": dict(result["train_eval"]["metrics"]),
                "val": dict(result["val_eval"]["metrics"]),
                "test": dict(result["test_eval"]["metrics"]),
            },
        }
        model_info["metrics"]["dataset_refresh"] = self._dataset_refresh_metadata()
        if self.ctx is not None:
            self.ctx.store.upsert_portfolio_training_run(
                run_id=str(result["run_id"]),
                run_type=run_type,
                model_version=str(result["model_version"]),
                status="completed",
                started_at=pd.Timestamp.utcnow().isoformat(),
                completed_at=pd.Timestamp.utcnow().isoformat(),
                algorithm="ppo",
                seed=self.config.seed,
                device=str(result["device"]),
                hyperparameters=self.config.to_dict(),
                metrics={
                    "train": dict(result["train_eval"]["metrics"]),
                    "val": dict(result["val_eval"]["metrics"]),
                    "test": dict(result["test_eval"]["metrics"]),
                },
                artifacts=dict(result.get("artifacts") or {}),
                notes=run_type,
            )
            for row in result.get("history", []):
                for metric_name, metric_value in row.items():
                    if metric_name == "update":
                        continue
                    self.ctx.store.upsert_portfolio_training_metric(
                        run_id=str(result["run_id"]),
                        step=int(row["update"]),
                        split="training",
                        metric_name=str(metric_name),
                        metric_value=float(metric_value),
                    )
            self.ctx.store.upsert_portfolio_model_info(**model_info)
            self.ctx.store.append_event(
                event_id=f"evt_{uuid4().hex[:10]}",
                event_type=EventType.PORTFOLIO_TRAINING_RUN,
                producer=self.agent_id,
                payload={
                    "run_id": str(result["run_id"]),
                    "run_type": run_type,
                    "model_version": str(result["model_version"]),
                    "metrics": model_info["metrics"],
                    "dataset_refresh": self._dataset_refresh_metadata(),
                },
                correlation_id=str(result["run_id"]),
            )
            self.ctx.store.append_event(
                event_id=f"evt_{uuid4().hex[:10]}",
                event_type=EventType.PORTFOLIO_MODEL_UPDATED,
                producer=self.agent_id,
                payload=model_info,
                correlation_id=str(result["run_id"]),
            )
            for snap in result.get("test_eval", {}).get("snapshots", []):
                rebalance_id = f"rb_{str(snap.get('rebalance_date')).replace(':', '').replace('-', '').replace(' ', '_')}"
                self.ctx.store.upsert_portfolio_rebalance_snapshot(
                    rebalance_id=rebalance_id,
                    rebalance_date=str(snap.get("rebalance_date") or ""),
                    model_version=str(result["model_version"]),
                    training_run_id=str(result["run_id"]),
                    fine_tune_run_id=str(result["run_id"]) if run_type == "fine_tune" else "",
                    active_traders=[],
                    selected_traders=list(snap.get("selected_traders") or []),
                    target_weights=dict(snap.get("target_weights") or {}),
                    target_cash_weight=float(snap.get("target_cash_weight") or 0.0),
                    diagnostics={
                        "n_active": int(snap.get("n_active") or 0),
                        "n_selected": int(snap.get("n_selected") or 0),
                    },
                    metadata={"source": "offline_test_eval"},
                )
            for item in result.get("forward_eval", []):
                evaluation_id = f"pfe_{uuid4().hex[:10]}"
                self.ctx.store.upsert_portfolio_forward_evaluation(
                    evaluation_id=evaluation_id,
                    rebalance_id=f"rb_{str(item.get('rebalance_date')).replace(':', '').replace('-', '').replace(' ', '_')}",
                    benchmark_name=str(item.get("benchmark_name") or "ppo"),
                    as_of=str(item.get("rebalance_date") or pd.Timestamp.utcnow().isoformat()),
                    cumulative_return_1y=float(item.get("cumulative_return_1y") or 0.0),
                    sharpe_1y=float(item.get("sharpe_1y") or 0.0),
                    max_drawdown_1y=float(item.get("max_drawdown_1y") or 0.0),
                    curve_points=list(item.get("curve_points") or []),
                    metadata={
                        "n_active": int(item.get("n_active") or 0),
                        "n_selected": int(item.get("n_selected") or 0),
                        "target_cash_weight": float(item.get("target_cash_weight") or 0.0),
                    },
                )
        self._latest_model_info = model_info
        self._latest_training_run = {
            "run_id": str(result["run_id"]),
            "run_type": run_type,
            "model_version": str(result["model_version"]),
        }
        return model_info

    def ensure_initial_model_ready(
        self,
        *,
        history_loader: Callable[[str], pd.DataFrame | pd.Series | None] | None,
    ) -> Dict[str, Any] | None:
        dataset = self._build_master_dataset(history_loader=history_loader)
        splits = self.dataset_builder.temporal_splits(dataset)
        latest_model = self._latest_model_from_store()
        needs_initial = latest_model is None or not str((latest_model or {}).get("checkpoint_path") or "").strip()

        if needs_initial:
            run_id = f"ppo_init_{uuid4().hex[:10]}"
            model_version = f"ppo_{pd.Timestamp.utcnow().strftime('%Y%m%d%H%M%S')}"
            result = self.trainer.train(
                dataset=dataset,
                splits=splits,
                run_id=run_id,
                model_version=model_version,
                run_type="initial_train",
                checkpoint_path=None,
            )
            return self._persist_training_result(run_type="initial_train", result=result)

        return latest_model

    def run_monthly_refresh_and_fine_tune(
        self,
        *,
        history_loader: Callable[[str], pd.DataFrame | pd.Series | None] | None,
        as_of: str | None = None,
        force: bool = False,
    ) -> Dict[str, Any] | None:
        dataset = self._build_master_dataset(history_loader=history_loader)
        splits = self.dataset_builder.temporal_splits(dataset)
        latest_model = self._latest_model_from_store()
        if latest_model is None or not str((latest_model or {}).get("checkpoint_path") or "").strip():
            return self.ensure_initial_model_ready(history_loader=history_loader)

        current_month = self._month_token(as_of)
        last_token = self._month_token(str(latest_model.get("fine_tuned_at") or latest_model.get("trained_at") or as_of))
        if (not force) and last_token == current_month:
            return latest_model

        run_id = f"ppo_ft_{uuid4().hex[:10]}"
        model_version = f"ppo_{pd.Timestamp.utcnow().strftime('%Y%m%d%H%M%S')}"
        result = self.trainer.train(
            dataset=dataset,
            splits=splits,
            run_id=run_id,
            model_version=model_version,
            run_type="fine_tune",
            checkpoint_path=str(latest_model["checkpoint_path"]),
        )
        return self._persist_training_result(run_type="fine_tune", result=result)

    def _build_training_comparison(self) -> pd.DataFrame:
        if self._latest_training_run is None or self.ctx is None:
            return pd.DataFrame()
        latest_rebalances = self.ctx.store.list_portfolio_forward_evaluations(limit=200)
        if not latest_rebalances:
            return pd.DataFrame()
        df = pd.DataFrame(latest_rebalances)
        if df.empty:
            return df
        summary = (
            df.groupby("benchmark_name", as_index=False)[["cumulative_return_1y", "sharpe_1y", "max_drawdown_1y"]]
            .mean(numeric_only=True)
            .rename(
                columns={
                    "benchmark_name": "portfolio",
                    "cumulative_return_1y": "forward_return_1y",
                    "sharpe_1y": "forward_sharpe_1y",
                    "max_drawdown_1y": "forward_maxdd_1y",
                }
            )
        )
        return summary

    def _pretty_plot_trader_name(self, trader_id: Any) -> str:
        txt = str(trader_id or "").strip()
        spec = self._promoted_specs.get(txt)
        if spec is not None:
            asset_txt = str(getattr(spec, "asset", "") or "").strip().upper()
            tf_txt = str(getattr(spec, "timeframe", "") or "").strip().upper()
            if asset_txt and tf_txt:
                return f"{asset_txt}_{tf_txt}"
        if txt.startswith("tr_"):
            parts = txt.split("_")
            if len(parts) >= 4:
                return f"{parts[1].upper()}_{parts[2].upper()}"
        return txt or "-"

    def _pretty_plot_label_map(self, labels: Sequence[Any]) -> list[str]:
        return [self._pretty_plot_trader_name(label) for label in labels]

    def _plot_training_dashboard(
        self,
        *,
        history_df: pd.DataFrame,
        train_curve: pd.DataFrame,
        val_curve: pd.DataFrame,
        test_curve: pd.DataFrame,
        forward_df: pd.DataFrame,
        weight_map: Mapping[str, float],
        euro_map: Mapping[str, float],
    ) -> Dict[str, Figure]:
        figs: Dict[str, Figure] = {}

        fig_reward, ax_reward = plt.subplots(figsize=(6, 2.8))
        if not history_df.empty and {"update", "average_reward"}.issubset(history_df.columns):
            ax_reward.plot(history_df["update"], history_df["average_reward"], color="tab:blue")
            ax_reward.set_title("Reward media PPO")
            ax_reward.grid(True, alpha=0.2)
        else:
            ax_reward.text(0.5, 0.5, "Sin historial PPO", ha="center", va="center")
            ax_reward.set_axis_off()
        figs["training_reward"] = fig_reward

        fig_loss, ax_loss = plt.subplots(figsize=(6, 2.8))
        if not history_df.empty:
            for key in ("policy_loss", "value_loss", "entropy"):
                if key in history_df.columns:
                    ax_loss.plot(history_df["update"], history_df[key], label=key)
            ax_loss.set_title("Curvas PPO")
            ax_loss.legend(fontsize=8)
            ax_loss.grid(True, alpha=0.2)
        else:
            ax_loss.text(0.5, 0.5, "Sin pérdidas PPO", ha="center", va="center")
            ax_loss.set_axis_off()
        figs["losses"] = fig_loss

        fig_curves, ax_curves = plt.subplots(figsize=(6, 3))
        plotted = False
        for label, df_curve in [("Train", train_curve), ("Val", val_curve), ("Test", test_curve)]:
            if isinstance(df_curve, pd.DataFrame) and not df_curve.empty and {"date", "equity"}.issubset(df_curve.columns):
                curve = df_curve.copy()
                curve["date"] = pd.to_datetime(curve["date"], errors="coerce")
                curve = curve.dropna(subset=["date"])
                ax_curves.plot(curve["date"], curve["equity"], label=label)
                plotted = True
        if plotted:
            ax_curves.set_title("P/L entrenamiento y validación PPO")
            ax_curves.legend(fontsize=8)
            ax_curves.grid(True, alpha=0.2)
            handles, labels = ax_curves.get_legend_handles_labels()
            pretty_labels = []
            for label in labels:
                if str(label).lower() in {"train", "val", "test"}:
                    pretty_labels.append(str(label))
                else:
                    pretty_labels.append(self._pretty_plot_trader_name(label))
            ax_curves.legend(handles, pretty_labels, fontsize=8)
        else:
            ax_curves.text(0.5, 0.5, "Sin curvas train/val/test", ha="center", va="center")
            ax_curves.set_axis_off()
        figs["rolling_curves"] = fig_curves

        fig_forward, ax_forward = plt.subplots(figsize=(6, 3))
        if isinstance(forward_df, pd.DataFrame) and not forward_df.empty:
            for benchmark, grp in forward_df.groupby("benchmark_name"):
                first_curve = grp.iloc[0].get("curve_points", [])
                if not first_curve:
                    continue
                curve_df = pd.DataFrame(first_curve)
                if curve_df.empty or "date" not in curve_df.columns:
                    continue
                curve_df["date"] = pd.to_datetime(curve_df["date"], errors="coerce")
                curve_df = curve_df.dropna(subset=["date"])
                ax_forward.plot(curve_df["date"], curve_df["equity"], label=str(benchmark))
            ax_forward.set_title("Forward 1Y por rebalance")
            ax_forward.legend(fontsize=8)
            ax_forward.grid(True, alpha=0.2)
        else:
            ax_forward.text(0.5, 0.5, "Sin curvas forward", ha="center", va="center")
            ax_forward.set_axis_off()
        figs["forward_curves"] = fig_forward

        fig_weights, ax_weights = plt.subplots(figsize=(6, 3))
        if euro_map:
            keys = list(euro_map.keys())
            vals = [float(euro_map[k]) for k in keys]
            plot_labels = self._pretty_plot_label_map(keys)
            ax_weights.bar(plot_labels, vals, color="tab:green")
            ax_weights.set_title("Asignación PPO en euros")
            ax_weights.tick_params(axis="x", rotation=90)
            ax_weights.tick_params(axis="x", labelsize=8)
        else:
            ax_weights.text(0.5, 0.5, "Sin pesos PPO", ha="center", va="center")
            ax_weights.set_axis_off()
        figs["weights_eur"] = fig_weights
        return figs

    def rebalance_active_signals(
        self,
        *,
        active_signals: Sequence[Mapping[str, Any]] | pd.DataFrame,
        total_capital_eur: float,
        frequency: str = "daily",
        lookback: int = 252,
        historical_pnl_paths: Mapping[str, str | Path] | None = None,
        historical_series_by_system: Mapping[str, pd.DataFrame | pd.Series] | None = None,
        history_loader: Callable[[str], pd.DataFrame | pd.Series | None] | None = None,
    ) -> Dict[str, Any]:
        if self.config.portfolio_manager_mode == "legacy":
            return self._rebalance_active_signals_legacy(
                active_signals=active_signals,
                total_capital_eur=total_capital_eur,
                frequency=frequency,
                lookback=lookback,
                historical_pnl_paths=historical_pnl_paths,
                historical_series_by_system=historical_series_by_system,
                history_loader=history_loader,
            )

        active_df = self.discover_active_systems(active_signals)
        if active_df.empty:
            return {
                "active_systems": active_df,
                "selected_tickers": [],
                "weights": {},
                "euros": {},
                "risk_contribution": {},
                "comparison": pd.DataFrame(),
                "returns_active": pd.DataFrame(),
                "returns_selected": pd.DataFrame(),
                "figures": {},
                "target_cash_weight": 1.0,
                "status": "no_active_signals",
            }

        try:
            model_info = self.ensure_initial_model_ready(history_loader=history_loader)
            if model_info is None or self._latest_dataset is None:
                raise ValueError("Portfolio PPO no tiene modelo listo.")
            previous_snapshot = self.ctx.store.get_latest_portfolio_rebalance_snapshot() if self.ctx is not None else None
            ppo_out = self.inference_service.infer(
                dataset=self._latest_dataset,
                checkpoint_path=str(model_info["checkpoint_path"]),
                active_trader_ids=active_df["trader_id"].tolist(),
                total_capital_eur=float(total_capital_eur),
                previous_snapshot=previous_snapshot,
            )
            ppo_out = self._enforce_min_open_positions(
                ppo_out=ppo_out,
                active_df=active_df,
                total_capital_eur=float(total_capital_eur),
            )
        except Exception as exc:
            emit_log(self.agent_id, "portfolio_ppo_fallback_legacy", console=False, error=str(exc))
            return self._rebalance_active_signals_legacy(
                active_signals=active_signals,
                total_capital_eur=total_capital_eur,
                frequency=frequency,
                lookback=lookback,
                historical_pnl_paths=historical_pnl_paths,
                historical_series_by_system=historical_series_by_system,
                history_loader=history_loader,
            )

        active_returns = pd.DataFrame(
            self._latest_dataset.returns,
            index=pd.Index(self._latest_dataset.dates),
            columns=self._latest_dataset.trader_ids,
        )[active_df["trader_id"].tolist()]
        selected_returns = active_returns[ppo_out["selected_tickers"]] if ppo_out["selected_tickers"] else pd.DataFrame(index=active_returns.index)
        selection_df = active_df.copy()
        selection_df["selected"] = selection_df["trader_id"].isin(set(ppo_out["selected_tickers"]))
        selection_df["weight"] = selection_df["trader_id"].map(ppo_out["weights"]).fillna(0.0)
        selection_df["euros"] = selection_df["trader_id"].map(ppo_out["euros"]).fillna(0.0)
        selection_df["risk_contribution"] = selection_df["weight"]

        latest_run = self._latest_training_run or {}
        history_rows = self.ctx.store.list_portfolio_training_metrics(str(latest_run.get("run_id"))) if (self.ctx is not None and latest_run.get("run_id")) else []
        history_df = pd.DataFrame(history_rows)
        if not history_df.empty:
            history_df = (
                history_df.pivot_table(index="step", columns="metric_name", values="metric_value", aggfunc="last")
                .reset_index()
                .rename(columns={"step": "update"})
            )
        train_curve = pd.read_csv(str(model_info["metrics"].get("train_curve_csv", ""))) if False else pd.DataFrame()
        artifacts = {}
        if latest_run.get("run_id") and self.ctx is not None:
            runs = {r["run_id"]: r for r in self.ctx.store.list_portfolio_training_runs(limit=20)}
            artifacts = dict((runs.get(str(latest_run["run_id"])) or {}).get("artifacts") or {})
        def _safe_csv(path_str: str) -> pd.DataFrame:
            try:
                if path_str:
                    return pd.read_csv(path_str)
            except Exception:
                pass
            return pd.DataFrame()
        train_curve = _safe_csv(str(artifacts.get("train_curve_csv", "")))
        val_curve = _safe_csv(str(artifacts.get("val_curve_csv", "")))
        test_curve = _safe_csv(str(artifacts.get("test_curve_csv", "")))
        forward_curve = pd.DataFrame(self.ctx.store.list_portfolio_forward_evaluations(limit=200)) if self.ctx is not None else pd.DataFrame()
        figures = self._plot_training_dashboard(
            history_df=history_df,
            train_curve=train_curve,
            val_curve=val_curve,
            test_curve=test_curve,
            forward_df=forward_curve,
            weight_map=ppo_out["weights"],
            euro_map=ppo_out["euros"],
        )

        rebalance_ts = pd.Timestamp.utcnow().isoformat()
        rebalance_id = f"rb_{rebalance_ts.replace(':', '').replace('-', '').replace(' ', '_')}"
        decision = PortfolioDecision(
            decision_id=rebalance_id,
            as_of=rebalance_ts,
            selected_traders=list(ppo_out["selected_tickers"]),
            weights={k: float(v) for k, v in ppo_out["weights"].items()},
            rationale="ppo policy over master universe with active mask",
            model_version=str(model_info["model_version"]),
            training_run_id=str(latest_run.get("run_id") or ""),
            fine_tune_run_id=str(latest_run.get("run_id") if latest_run.get("run_type") == "fine_tune" else ""),
            target_cash_weight=float(ppo_out["target_cash_weight"]),
            active_universe_size=int(ppo_out["active_universe_size"]),
            selected_universe_size=int(ppo_out["selected_universe_size"]),
            metadata={
                **dict(ppo_out.get("diagnostics") or {}),
                "dataset_refresh": self._dataset_refresh_metadata(),
            },
        )

        if self.ctx is not None:
            self.ctx.store.set_agent_status(self.agent_id, AgentStatus.RUNNING, "ppo portfolio inference")
            self.ctx.store.append_event(
                event_id=f"evt_{uuid4().hex[:10]}",
                event_type=EventType.PORTFOLIO_DECISION,
                producer=self.agent_id,
                payload=decision.to_dict(),
                correlation_id=decision.decision_id,
            )
            self.ctx.store.upsert_portfolio_rebalance_snapshot(
                rebalance_id=rebalance_id,
                rebalance_date=decision.as_of,
                model_version=str(model_info["model_version"]),
                training_run_id=str(latest_run.get("run_id") or ""),
                fine_tune_run_id=str(latest_run.get("run_id") if latest_run.get("run_type") == "fine_tune" else ""),
                active_traders=active_df["trader_id"].astype(str).tolist(),
                selected_traders=list(ppo_out["selected_tickers"]),
                target_weights={k: float(v) for k, v in ppo_out["weights"].items()},
                target_cash_weight=float(ppo_out["target_cash_weight"]),
                diagnostics=dict(ppo_out.get("diagnostics") or {}),
                metadata={
                    "symbols": active_df["symbol"].astype(str).tolist(),
                    "euros": {k: float(v) for k, v in ppo_out["euros"].items()},
                    "dataset_refresh": self._dataset_refresh_metadata(),
                },
            )
            for idx, row in enumerate(self.ctx.store.list_portfolio_forward_evaluations(limit=500)):
                _ = idx
            emit_log(self.agent_id, "portfolio_decision", console=False, decision=decision.to_dict())
            self.ctx.store.set_agent_status(self.agent_id, AgentStatus.IDLE, "ppo portfolio inference done")

        return {
            "active_systems": active_df,
            "selected_tickers": list(ppo_out["selected_tickers"]),
            "weights": dict(ppo_out["weights"]),
            "euros": dict(ppo_out["euros"]),
            "risk_contribution": dict(ppo_out["weights"]),
            "comparison": self._build_training_comparison(),
            "returns_active": active_returns,
            "returns_selected": selected_returns,
            "selection_df": selection_df,
            "figures": figures,
            "decision": decision.to_dict(),
            "target_cash_weight": float(ppo_out["target_cash_weight"]),
            "model_version": str(model_info["model_version"]),
            "training_run_id": str(latest_run.get("run_id") or ""),
            "status": "ppo_ready",
            "diagnostics": {
                **dict(ppo_out.get("diagnostics") or {}),
                "dataset_refresh": self._dataset_refresh_metadata(),
            },
        }

    def _rebalance_active_signals_legacy(
        self,
        *,
        active_signals: Sequence[Mapping[str, Any]] | pd.DataFrame,
        total_capital_eur: float,
        frequency: str = "daily",
        lookback: int = 252,
        historical_pnl_paths: Mapping[str, str | Path] | None = None,
        historical_series_by_system: Mapping[str, pd.DataFrame | pd.Series] | None = None,
        history_loader: Callable[[str], pd.DataFrame | pd.Series | None] | None = None,
    ) -> Dict[str, Any]:
        active_df = self.discover_active_systems(active_signals)
        if active_df.empty:
            return {
                "active_systems": active_df,
                "selected_tickers": [],
                "weights": {},
                "euros": {},
                "risk_contribution": {},
                "comparison": pd.DataFrame(),
                "returns_active": pd.DataFrame(),
                "returns_selected": pd.DataFrame(),
                "condition_evolution": [],
                "best_cov_matrix": pd.DataFrame(),
                "best_corr_matrix": pd.DataFrame(),
                "figures": {},
            }

        returns_active = self.load_system_returns(
            active_df,
            historical_pnl_paths=historical_pnl_paths,
            historical_series_by_system=historical_series_by_system,
            history_loader=history_loader,
            frequency=frequency,
            lookback=lookback,
        )
        if returns_active.empty:
            selected_df = pd.DataFrame(index=active_df["trader_id"].tolist())
            return {
                "active_systems": active_df,
                "selected_tickers": [],
                "weights": {},
                "euros": {},
                "risk_contribution": {},
                "comparison": pd.DataFrame(),
                "returns_active": returns_active,
                "returns_selected": pd.DataFrame(),
                "condition_evolution": [],
                "best_cov_matrix": pd.DataFrame(),
                "best_corr_matrix": pd.DataFrame(),
                "selection_df": selected_df,
                "figures": {},
            }

        genetic = self.genetic_select_universe(returns_active)
        returns_selected = returns_active[genetic.selected_tickers] if genetic.selected_tickers else pd.DataFrame(index=returns_active.index)
        hrp_all = self.allocate_weights_hrp(returns_active, total_capital_eur=total_capital_eur)
        hrp_selected = self.allocate_weights_hrp(returns_selected, total_capital_eur=total_capital_eur)
        equal_selected = self.allocate_weights_equal(returns_selected, total_capital_eur=total_capital_eur)
        comparison = self.compare_selected_vs_all_active(
            active_returns=returns_active,
            selected_returns=returns_selected,
            hrp_all=hrp_all,
            hrp_selected=hrp_selected,
            equal_selected=equal_selected,
        )

        selection_df = active_df.copy()
        selection_df["selected"] = selection_df["trader_id"].isin(set(genetic.selected_tickers))
        selection_df["weight"] = selection_df["trader_id"].map(hrp_selected.get("weights", {})).fillna(0.0)
        selection_df["euros"] = selection_df["trader_id"].map(hrp_selected.get("euros", {})).fillna(0.0)
        selection_df["risk_contribution"] = selection_df["trader_id"].map(hrp_selected.get("risk_contribution", {})).fillna(0.0)

        rolling_curves = None
        weighted_curve = None
        if not returns_selected.empty:
            rolling_curves = ((1.0 + returns_selected).cumprod() - 1.0)
            sel_weights = pd.Series(hrp_selected.get("weights", {}), dtype=float).reindex(returns_selected.columns).fillna(0.0)
            if float(sel_weights.sum()) > 0:
                sel_weights = sel_weights / sel_weights.sum()
                weighted_curve = ((1.0 + (returns_selected.mul(sel_weights, axis=1).sum(axis=1))).cumprod() - 1.0)

        figures = self.plot_portfolio_manager_dashboard(
            condition_evolution=genetic.condition_evolution,
            corr_selected=genetic.best_corr_matrix,
            linkage_matrix=hrp_selected.get("linkage"),
            weights_eur=hrp_selected.get("euros", {}),
            risk_contribution=hrp_selected.get("risk_contribution", {}),
            comparison_df=comparison,
            rolling_curves=rolling_curves,
            weighted_curve=weighted_curve,
        )

        out = {
            "active_systems": active_df,
            "selected_tickers": genetic.selected_tickers,
            "weights": hrp_selected.get("weights", {}),
            "euros": hrp_selected.get("euros", {}),
            "risk_contribution": hrp_selected.get("risk_contribution", {}),
            "comparison": comparison,
            "returns_active": returns_active,
            "returns_selected": returns_selected,
            "condition_evolution": genetic.condition_evolution,
            "best_cov_matrix": genetic.best_cov_matrix,
            "best_corr_matrix": genetic.best_corr_matrix,
            "selection_df": selection_df,
            "figures": figures,
            "hrp_all": hrp_all,
            "hrp_selected": hrp_selected,
            "equal_selected": equal_selected,
        }

        if self.ctx is not None:
            decision = PortfolioDecision(
                decision_id=f"pm_{uuid4().hex[:10]}",
                as_of=pd.Timestamp.utcnow().isoformat(),
                selected_traders=list(genetic.selected_tickers),
                weights={k: float(v) for k, v in hrp_selected.get("weights", {}).items()},
                rationale="genetic selection by condition number + HRP weights",
            )
            self.ctx.store.set_agent_status(self.agent_id, AgentStatus.RUNNING, "portfolio optimization")
            self.ctx.store.append_event(
                event_id=f"evt_{uuid4().hex[:10]}",
                event_type=EventType.PORTFOLIO_DECISION,
                producer=self.agent_id,
                payload=decision.to_dict(),
                correlation_id=decision.decision_id,
            )
            emit_log(self.agent_id, "portfolio_decision", console=False, decision=decision.to_dict())
            self.ctx.store.set_agent_status(self.agent_id, AgentStatus.IDLE, "portfolio optimization done")
            out["decision"] = decision.to_dict()
        return out

    # Compatibilidad con chequeos antiguos.
    def rebalance(self, *, as_of: str, max_weight: float = 0.6, min_score: float = -0.25) -> PortfolioDecision:
        _ = (max_weight, min_score)
        decision = PortfolioDecision(
            decision_id=f"pm_{uuid4().hex[:10]}",
            as_of=as_of,
            selected_traders=[],
            weights={},
            rationale="legacy rebalance disabled; use rebalance_active_signals",
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

    def get_broker_positions(self) -> list[dict]:
        if self.ctx is None or self.ctx.execution_router is None:
            return []
        return self.ctx.execution_router.get_open_positions(actor=self.agent_id)

    def get_market_snapshot(self, symbol: str) -> dict:
        if self.ctx is None or self.ctx.execution_router is None:
            return {}
        return self.ctx.execution_router.get_market_snapshot(actor=self.agent_id, symbol=symbol)

