"""Tests del optimizador hibrido GA + PSO del PortfolioManagerProcess."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.agents.portfolio_manager_process import PortfolioManagerProcess
from app.contracts import PortfolioDecision
from app.services.portfolio_optimizer import (
    PortfolioOptimizerConfig,
    compute_corr_media,
    compute_fitness,
    compute_mdd,
    compute_sharpe_neto,
    genetic_select_subsets,
    optimize_portfolio_ga_pso,
    pso_optimize_weights,
    repair_chromosome,
    repair_weights,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_returns(seed: int = 0, n_traders: int = 12, n_weeks: int = 120) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=0.001, scale=0.02, size=(n_weeks, n_traders))
    # Anyade un poco de estructura: tres clusters correlacionados.
    cluster_a = rng.normal(loc=0.0, scale=0.01, size=(n_weeks, 1))
    cluster_b = rng.normal(loc=0.0, scale=0.01, size=(n_weeks, 1))
    base[:, :4] += cluster_a
    base[:, 4:8] += cluster_b
    return base


def _basic_config(**overrides: object) -> PortfolioOptimizerConfig:
    defaults = dict(
        min_selected_traders=3,
        max_selected_traders=8,
        max_weight_per_trader=0.5,
        max_cash_weight=0.5,
        min_live_weight=0.0,
        ga_population_size=20,
        ga_generations=5,
        ga_early_stopping_generations=3,
        ga_tournament_size=3,
        ga_crossover_rate=0.8,
        ga_mutation_rate=0.05,
        ga_elitism=2,
        top_k_subsets_for_pso=2,
        pso_swarm_size=12,
        pso_iterations=10,
        pso_early_stopping_iterations=5,
        random_seed=123,
    )
    defaults.update(overrides)
    return PortfolioOptimizerConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Fitness behavior
# ---------------------------------------------------------------------------


def test_fitness_increases_when_sharpe_increases() -> None:
    f_low = compute_fitness(0.5, 0.10, 0.20, lambda_dd=1.0, lambda_corr=0.5)
    f_high = compute_fitness(1.0, 0.10, 0.20, lambda_dd=1.0, lambda_corr=0.5)
    assert f_high > f_low


def test_fitness_decreases_when_mdd_increases() -> None:
    f_low_dd = compute_fitness(0.8, 0.10, 0.20, lambda_dd=1.0, lambda_corr=0.5)
    f_high_dd = compute_fitness(0.8, 0.40, 0.20, lambda_dd=1.0, lambda_corr=0.5)
    assert f_low_dd > f_high_dd


def test_fitness_decreases_when_corr_increases() -> None:
    f_low_c = compute_fitness(0.8, 0.10, 0.10, lambda_dd=1.0, lambda_corr=0.5)
    f_high_c = compute_fitness(0.8, 0.10, 0.50, lambda_dd=1.0, lambda_corr=0.5)
    assert f_low_c > f_high_c


def test_compute_sharpe_neto_handles_constant_series() -> None:
    arr = np.zeros(50, dtype=float)
    assert compute_sharpe_neto(arr) == 0.0


def test_compute_mdd_returns_positive_drawdown() -> None:
    returns = np.array([0.05, -0.10, -0.05, 0.02, 0.04])
    mdd = compute_mdd(returns)
    assert mdd > 0.0
    assert mdd <= 1.0


def test_compute_corr_media_runs_with_two_assets() -> None:
    rng = np.random.default_rng(0)
    rets = rng.normal(size=(60, 3))
    corr = compute_corr_media(rets, [0, 1])
    assert 0.0 <= corr <= 1.0


# ---------------------------------------------------------------------------
# Repair logic (chromosome and weights)
# ---------------------------------------------------------------------------


def test_repair_chromosome_enforces_min_max_selected() -> None:
    rng = np.random.default_rng(0)
    chrom = np.zeros(20, dtype=int)
    repaired = repair_chromosome(chrom, min_selected=5, max_selected=10, rng=rng)
    assert 5 <= repaired.sum() <= 10
    too_many = np.ones(20, dtype=int)
    repaired2 = repair_chromosome(too_many, min_selected=5, max_selected=10, rng=rng)
    assert 5 <= repaired2.sum() <= 10


def test_repair_weights_respects_caps_and_simplex() -> None:
    weights = np.array([0.6, 0.3, 0.5, 0.0, 0.1, 0.2])  # 5 traders + cash
    repaired = repair_weights(
        weights,
        max_weight_per_trader=0.30,
        max_cash_weight=0.40,
        min_live_weight=0.0,
    )
    assert (repaired >= 0.0).all()
    assert pytest.approx(float(repaired.sum()), rel=1e-6) == 1.0
    # Cap por trader (todos los componentes salvo el ultimo)
    assert (repaired[:-1] <= 0.30 + 1e-9).all()
    # Cap de cash
    assert repaired[-1] <= 0.40 + 1e-9


def test_repair_weights_handles_all_zero_falls_to_cash() -> None:
    weights = np.zeros(4)
    repaired = repair_weights(
        weights,
        max_weight_per_trader=0.5,
        max_cash_weight=1.0,
        min_live_weight=0.0,
    )
    assert pytest.approx(float(repaired.sum()), rel=1e-6) == 1.0
    assert repaired[-1] == pytest.approx(1.0)


def test_repair_weights_drops_below_min_live_weight() -> None:
    weights = np.array([0.005, 0.4, 0.5, 0.095])  # 3 traders + cash; primero < min
    repaired = repair_weights(
        weights,
        max_weight_per_trader=0.6,
        max_cash_weight=0.5,
        min_live_weight=0.02,
    )
    assert pytest.approx(float(repaired.sum()), rel=1e-6) == 1.0
    # El primero debe haberse anulado por estar por debajo del minimo.
    assert repaired[0] == 0.0 or repaired[0] >= 0.02


# ---------------------------------------------------------------------------
# Genetic selection
# ---------------------------------------------------------------------------


def test_genetic_select_subsets_returns_valid_chromosomes() -> None:
    returns = _make_returns(seed=1, n_traders=10, n_weeks=80)
    config = _basic_config()
    subsets = genetic_select_subsets(returns, config=config)
    assert subsets, "El GA debe devolver al menos un subconjunto"
    for s in subsets:
        n = int(s.chromosome.sum())
        assert config.min_selected_traders <= n <= config.max_selected_traders
        assert len(s.indices) == n


# ---------------------------------------------------------------------------
# PSO optimization
# ---------------------------------------------------------------------------


def test_pso_optimize_weights_returns_valid_weights() -> None:
    returns = _make_returns(seed=2, n_traders=5, n_weeks=80)
    config = _basic_config(max_weight_per_trader=0.4, max_cash_weight=0.3)
    result = pso_optimize_weights(returns, config=config)
    weights = result.weights
    assert weights.size == returns.shape[1] + 1
    assert (weights >= 0.0).all()
    assert pytest.approx(float(weights.sum()), rel=1e-6) == 1.0
    assert (weights[:-1] <= 0.40 + 1e-6).all()
    assert weights[-1] <= 0.30 + 1e-6


# ---------------------------------------------------------------------------
# End-to-end orchestrator
# ---------------------------------------------------------------------------


def test_optimize_portfolio_ga_pso_full_pipeline() -> None:
    returns = _make_returns(seed=3, n_traders=12, n_weeks=104)
    config = _basic_config(min_selected_traders=4, max_selected_traders=8)
    result = optimize_portfolio_ga_pso(returns, config=config)
    assert result.status == "ok"
    assert 4 <= len(result.selected_indices) <= 8
    assert (result.weights >= 0.0).all()
    assert result.cash_weight >= 0.0
    total = float(result.weights.sum() + result.cash_weight)
    assert pytest.approx(total, rel=1e-6) == 1.0
    assert (result.weights <= config.max_weight_per_trader + 1e-6).all()
    assert result.cash_weight <= config.max_cash_weight + 1e-6


def test_optimize_portfolio_handles_too_few_traders() -> None:
    returns = _make_returns(seed=4, n_traders=2, n_weeks=80)
    config = _basic_config(min_selected_traders=5, max_selected_traders=10)
    result = optimize_portfolio_ga_pso(returns, config=config)
    assert result.status == "degraded_few_traders"
    # Debe seguir cumpliendo el simplex.
    total = float(np.sum(result.weights) + result.cash_weight)
    assert pytest.approx(total, rel=1e-6) == 1.0


# ---------------------------------------------------------------------------
# PortfolioManagerProcess contract
# ---------------------------------------------------------------------------


def _build_history_loader(returns_df: pd.DataFrame):
    def _loader(trader_id: str):
        if trader_id not in returns_df.columns:
            return None
        series = returns_df[trader_id]
        return series.to_frame(name="weekly_return")

    return _loader


def test_portfolio_manager_decision_contract() -> None:
    n_traders = 10
    n_weeks = 110
    rng = np.random.default_rng(7)
    dates = pd.date_range("2022-01-07", periods=n_weeks, freq="W-FRI")
    returns_matrix = rng.normal(loc=0.001, scale=0.015, size=(n_weeks, n_traders))
    trader_ids = [f"tr_{i:02d}" for i in range(n_traders)]
    returns_df = pd.DataFrame(returns_matrix, index=dates, columns=trader_ids)

    config = PortfolioOptimizerConfig(
        min_selected_traders=3,
        max_selected_traders=6,
        max_weight_per_trader=0.4,
        max_cash_weight=0.4,
        min_live_weight=0.0,
        ga_population_size=20,
        ga_generations=4,
        ga_early_stopping_generations=2,
        ga_elitism=2,
        top_k_subsets_for_pso=2,
        pso_swarm_size=10,
        pso_iterations=8,
        pso_early_stopping_iterations=3,
        random_seed=7,
        lookback_weeks=104,
    )
    process = PortfolioManagerProcess(ctx=None, optimizer_config=config)
    active_signals = [
        {"trader_id": tid, "symbol": f"SYM{idx}", "side": "buy"}
        for idx, tid in enumerate(trader_ids)
    ]
    out = process.rebalance_active_signals(
        active_signals=active_signals,
        total_capital_eur=100_000.0,
        history_loader=_build_history_loader(returns_df),
    )

    assert out["optimizer_mode"] == "ga_pso"
    assert out["status"] == "ok"
    assert isinstance(out["decision"], dict)
    decision = out["decision"]
    for field in (
        "decision_id",
        "as_of",
        "selected_traders",
        "weights",
        "target_cash_weight",
        "active_universe_size",
        "valid_universe_size",
        "selected_universe_size",
        "fitness",
        "sharpe_neto",
        "mdd",
        "corr_media",
        "metadata",
    ):
        assert field in decision, f"Falta campo {field} en la decision"
    assert decision["optimizer_mode"] == "ga_pso"
    assert bool(decision["metadata"].get("ppo_removed"))
    weights_total = sum(float(v) for v in decision["weights"].values()) + float(decision["target_cash_weight"])
    assert pytest.approx(weights_total, rel=1e-6) == 1.0
    assert (np.array(list(decision["weights"].values())) <= config.max_weight_per_trader + 1e-6).all()
    assert decision["target_cash_weight"] <= config.max_cash_weight + 1e-6


def test_portfolio_manager_handles_no_signals() -> None:
    process = PortfolioManagerProcess(ctx=None, optimizer_config=_basic_config())
    out = process.rebalance_active_signals(
        active_signals=[],
        total_capital_eur=100_000.0,
        history_loader=lambda _tid: None,
    )
    assert out["status"] == "no_active_signals"
    assert out["selected_tickers"] == []
    assert out["target_cash_weight"] == pytest.approx(1.0)


def test_portfolio_manager_handles_no_history() -> None:
    process = PortfolioManagerProcess(ctx=None, optimizer_config=_basic_config())
    active_signals = [{"trader_id": "tr_a", "symbol": "AAA", "side": "buy"}]
    out = process.rebalance_active_signals(
        active_signals=active_signals,
        total_capital_eur=100_000.0,
        history_loader=lambda _tid: None,
    )
    assert out["status"] == "no_history"
    assert out["selected_tickers"] == []


def test_portfolio_decision_dataclass_has_no_ppo_fields() -> None:
    """Garantiza que PortfolioDecision ya no expone campos PPO."""
    decision = PortfolioDecision(
        decision_id="d1",
        as_of="2026-05-01T00:00:00Z",
        selected_traders=["tr_1"],
        weights={"tr_1": 1.0},
    )
    payload = decision.to_dict()
    for forbidden in ("model_version", "training_run_id", "fine_tune_run_id"):
        assert forbidden not in payload, f"PortfolioDecision sigue exponiendo {forbidden}"


def test_portfolio_support_module_no_longer_exports_ppo_components() -> None:
    """El paquete `portfolio_support` (antes `portfolio_rl`) no debe exportar PPO."""
    import app.services.portfolio_support as pkg

    forbidden_attrs = {
        "PPOTrainer",
        "PPOInferenceService",
        "MaskedPortfolioPolicy",
        "WeeklyPortfolioEnv",
        "PortfolioDatasetBuilder",
        "PortfolioDataset",
        "PortfolioArtifactsManager",
        "PPOPortfolioConfig",
        "PortfolioPolicyEvaluator",
        "build_weekly_feature_dataset",
    }
    exposed = set(getattr(pkg, "__all__", []))
    leaked = exposed & forbidden_attrs
    assert not leaked, f"portfolio_support sigue exportando componentes PPO: {leaked}"


def test_portfolio_optimizer_files_present_and_no_ppo_files() -> None:
    """Verifica el plan de archivos: portfolio_optimizer existe; los PPO no."""
    repo = Path(__file__).resolve().parents[2]
    assert (repo / "app" / "services" / "portfolio_optimizer.py").exists()
    services_dir = repo / "app" / "services"
    for ppo_name in (
        "ppo_trainer.py",
        "policy.py",
        "env.py",
        "evaluator.py",
        "feature_builder.py",
        "inference.py",
        "dataset_builder.py",
        "artifacts.py",
    ):
        assert not (services_dir / "portfolio_support" / ppo_name).exists(), (
            f"Sigue existiendo el archivo PPO {ppo_name} en portfolio_support/"
        )
        assert not (services_dir / "portfolio_rl" / ppo_name).exists(), (
            f"Sigue existiendo el archivo PPO {ppo_name} en portfolio_rl/ legacy"
        )
    assert not (services_dir / "portfolio_rl").exists(), (
        "El paquete legacy `portfolio_rl/` no debe seguir presente"
    )
