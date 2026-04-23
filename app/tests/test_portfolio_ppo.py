from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

from app.agents.portfolio_agent import PortfolioManagerAgent
from app.contracts import PromotedTraderSpec
from app.runtime.development_operational_supervisor import DevelopmentOperationalSupervisor
from app.services.portfolio_rl import (
    PPOInferenceService,
    PPOPortfolioConfig,
    PPOTrainer,
    PortfolioArtifactsManager,
    PortfolioDatasetBuilder,
    UniverseRegistry,
    build_weekly_feature_dataset,
)
from app.storage import StateStore


def _tmp_db_path() -> Path:
    base = Path("app/.tmp/tests")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"portfolio_state_{uuid4().hex[:8]}.sqlite"


def _tmp_artifact_root() -> str:
    base = Path("app/.tmp/tests") / f"portfolio_rl_{uuid4().hex[:8]}"
    base.mkdir(parents=True, exist_ok=True)
    return str(base)


def _tmp_dir() -> Path:
    base = Path("app/.tmp/tests") / f"portfolio_refresh_{uuid4().hex[:8]}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _synthetic_dataset():
    dates = pd.date_range("2022-01-07", periods=36, freq="W-FRI")
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        {
            "tr_A": 0.002 + rng.normal(0, 0.01, size=len(dates)),
            "tr_B": 0.001 + rng.normal(0, 0.008, size=len(dates)),
            "tr_C": 0.0005 + rng.normal(0, 0.006, size=len(dates)),
        },
        index=dates,
    )
    active = pd.DataFrame(
        {
            "tr_A": 1.0,
            "tr_B": [1.0] * 24 + [0.0] * 12,
            "tr_C": [0.0] * 4 + [1.0] * 32,
        },
        index=dates,
    )
    info = {
        "tr_A": {"promotion_date": str(dates[0]), "trade_count": 40, "avg_trade_duration_days": 7.0, "win_rate_pct": 58.0},
        "tr_B": {"promotion_date": str(dates[2]), "trade_count": 22, "avg_trade_duration_days": 9.0, "win_rate_pct": 51.0},
        "tr_C": {"promotion_date": str(dates[6]), "trade_count": 18, "avg_trade_duration_days": 12.0, "win_rate_pct": 55.0},
    }
    dataset = build_weekly_feature_dataset(returns, active, info)
    splits = {"train": slice(0, 24), "val": slice(24, 30), "test": slice(30, 36)}
    return dataset, splits


def test_policy_masks_inactive_and_sums_to_one() -> None:
    dataset, _ = _synthetic_dataset()
    config = PPOPortfolioConfig(device_preference="cpu", artifact_root=_tmp_artifact_root(), max_weight_per_trader=0.6)
    artifacts = PortfolioArtifactsManager(config)
    trainer = PPOTrainer(config, artifacts)
    policy = trainer.build_policy(dataset)

    trader = torch.tensor(dataset.trader_features[10], dtype=torch.float32).unsqueeze(0)
    dynamic = torch.zeros((1, dataset.n_traders, 4), dtype=torch.float32)
    trader = torch.cat([trader, dynamic], dim=-1)
    global_f = torch.tensor(dataset.global_features[10], dtype=torch.float32).unsqueeze(0)
    global_f = torch.cat([global_f, torch.zeros((1, 7), dtype=torch.float32)], dim=-1)
    active = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)

    out = policy.sample_action(
        trader,
        global_f,
        active,
        max_weight_per_trader=config.max_weight_per_trader,
        deterministic=True,
    )
    weights = out["weights"].squeeze(0).detach().cpu().numpy()
    assert abs(float(weights.sum()) - 1.0) < 1e-5
    assert float(weights[1]) == 0.0
    assert float(weights[-1]) >= 0.0


def test_universe_registry_supports_new_traders() -> None:
    db_path = _tmp_db_path()
    if db_path.exists():
        db_path.unlink()
    store = StateStore(db_path=db_path)
    registry = UniverseRegistry(store)
    specs = {
        "tr_A": PromotedTraderSpec(trader_id="tr_A", asset="AAPL", timeframe="D1", long_rules=["r1"], short_rules=[], origin_experiment_id="exp1"),
        "tr_B": PromotedTraderSpec(trader_id="tr_B", asset="MSFT", timeframe="D1", long_rules=["r2"], short_rules=[], origin_experiment_id="exp2"),
    }
    registry.sync_promoted_specs(specs)
    members = registry.list_members()
    assert set(members.keys()) == {"tr_A", "tr_B"}

    specs["tr_C"] = PromotedTraderSpec(trader_id="tr_C", asset="NVDA", timeframe="D1", long_rules=["r3"], short_rules=[], origin_experiment_id="exp3")
    registry.sync_promoted_specs(specs)
    members = registry.list_members()
    assert "tr_C" in members


def test_training_and_inference_roundtrip() -> None:
    dataset, splits = _synthetic_dataset()
    config = PPOPortfolioConfig(
        device_preference="cpu",
        artifact_root=_tmp_artifact_root(),
        max_updates_initial=2,
        ppo_epochs=2,
        batch_size=8,
        hidden_dim_encoder=16,
        hidden_dim_head=32,
    )
    artifacts = PortfolioArtifactsManager(config)
    trainer = PPOTrainer(config, artifacts)
    result = trainer.train(
        dataset=dataset,
        splits=splits,
        run_id=f"run_{uuid4().hex[:8]}",
        model_version=f"ppo_{uuid4().hex[:8]}",
        run_type="initial_train",
    )
    assert Path(result["checkpoint_path"]).exists()
    assert result["forward_eval"]

    inference = PPOInferenceService(config, artifacts)
    inferred = inference.infer(
        dataset=dataset,
        checkpoint_path=result["checkpoint_path"],
        active_trader_ids=["tr_A", "tr_C"],
        total_capital_eur=100000.0,
    )
    total = sum(inferred["weights"].values()) + float(inferred["target_cash_weight"])
    assert abs(total - 1.0) < 1e-5
    assert "tr_B" not in inferred["weights"]


def test_portfolio_manager_enforces_min_open_positions_with_ten_or_more_signals() -> None:
    agent = PortfolioManagerAgent()
    active_df = pd.DataFrame({"trader_id": [f"tr_{idx:02d}" for idx in range(12)]})
    out = agent._enforce_min_open_positions(
        ppo_out={
            "selected_tickers": [],
            "weights": {"tr_00": 1e-9},
            "euros": {"tr_00": 0.0001},
            "target_cash_weight": 1.0,
            "selected_universe_size": 0,
            "diagnostics": {},
        },
        active_df=active_df,
        total_capital_eur=100000.0,
    )
    assert len(out["selected_tickers"]) == 10
    assert abs(sum(out["weights"].values()) - 1.0) < 1e-9
    assert abs(float(out["target_cash_weight"])) < 1e-9
    assert all(abs(float(v) - 0.1) < 1e-9 for v in out["weights"].values())


def test_portfolio_manager_enforces_all_signals_when_active_set_is_below_minimum() -> None:
    agent = PortfolioManagerAgent()
    active_df = pd.DataFrame({"trader_id": [f"tr_{idx:02d}" for idx in range(6)]})
    out = agent._enforce_min_open_positions(
        ppo_out={
            "selected_tickers": [],
            "weights": {"tr_00": 1e-12},
            "euros": {"tr_00": 0.0},
            "target_cash_weight": 1.0,
            "selected_universe_size": 0,
            "diagnostics": {},
        },
        active_df=active_df,
        total_capital_eur=100000.0,
    )
    assert len(out["selected_tickers"]) == 6
    assert set(out["selected_tickers"]) == set(active_df["trader_id"].tolist())
    assert all(abs(float(v) - 0.15) < 1e-9 for v in out["weights"].values())
    assert abs(float(out["target_cash_weight"]) - 0.1) < 1e-9


def test_fine_tune_loads_existing_checkpoint() -> None:
    dataset, splits = _synthetic_dataset()
    config = PPOPortfolioConfig(
        device_preference="cpu",
        artifact_root=_tmp_artifact_root(),
        max_updates_initial=1,
        max_updates_fine_tune=1,
        ppo_epochs=1,
        batch_size=8,
        hidden_dim_encoder=16,
        hidden_dim_head=32,
    )
    artifacts = PortfolioArtifactsManager(config)
    trainer = PPOTrainer(config, artifacts)
    initial = trainer.train(
        dataset=dataset,
        splits=splits,
        run_id=f"run_{uuid4().hex[:8]}",
        model_version=f"ppo_{uuid4().hex[:8]}",
        run_type="initial_train",
    )
    fine_tuned = trainer.train(
        dataset=dataset,
        splits=splits,
        run_id=f"run_{uuid4().hex[:8]}",
        model_version=f"ppo_{uuid4().hex[:8]}",
        run_type="fine_tune",
        checkpoint_path=initial["checkpoint_path"],
    )
    assert Path(fine_tuned["checkpoint_path"]).exists()


def test_dataset_builder_prefers_real_weekly_mask_from_refresh_artifacts() -> None:
    db_path = _tmp_db_path()
    store = StateStore(db_path=db_path)
    root = _tmp_dir()
    trader_id = "tr_A"
    run_id = f"bt_{uuid4().hex[:8]}"
    dates = pd.date_range("2024-01-05", periods=60, freq="W-FRI")
    weekly_returns = pd.DataFrame(
        {
            "week_end": dates.date.astype(str),
            "weekly_return": np.linspace(-0.01, 0.02, len(dates)),
            "equity_close": 10000.0 * np.cumprod(1.0 + np.linspace(-0.01, 0.02, len(dates))),
            "balance_close": 10000.0 * np.cumprod(1.0 + np.linspace(-0.01, 0.02, len(dates))),
        }
    )
    weekly_mask = pd.DataFrame(
        {
            "week_end": dates.date.astype(str),
            "active": ([0] * 8) + ([1] * 20) + ([0] * 32),
            "side": ["buy"] * len(dates),
            "bars_in_market": [1] * len(dates),
            "pnl_week": np.linspace(-0.01, 0.02, len(dates)),
            "mask_source": ["real_backtest"] * len(dates),
        }
    )
    pnl_path = root / "historical_pnl.csv"
    trades_path = root / "historical_trades.csv"
    weekly_returns_path = root / "weekly_returns.csv"
    weekly_mask_path = root / "weekly_signal_mask.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=200, freq="B"),
            "equity": 10000.0 * np.cumprod(1.0 + np.random.default_rng(11).normal(0.0005, 0.005, 200)),
            "balance": 10000.0 * np.cumprod(1.0 + np.random.default_rng(12).normal(0.0004, 0.004, 200)),
        }
    ).to_csv(pnl_path, index=False)
    pd.DataFrame({"entry_time": ["2024-02-01"], "exit_time": ["2024-02-09"], "profit": [12.0], "side": ["buy"]}).to_csv(trades_path, index=False)
    weekly_returns.to_csv(weekly_returns_path, index=False)
    weekly_mask.to_csv(weekly_mask_path, index=False)
    store.upsert_trader_backtest_run(
        run_id=run_id,
        trader_id=trader_id,
        asset="AAPL",
        timeframe="D1",
        start_date="2024-01-01",
        end_date=str(dates[-1].date()),
        cutoff_date=str(dates[-1].date()),
        rules_hash="rules_hash",
        price_data_fingerprint="fingerprint",
        status="completed",
        summary={"n_trades": 12, "trade_stats": {"avg_trade_duration_days": 6.0, "win_rate_pct": 55.0}},
    )
    store.upsert_trader_backtest_artifacts(
        run_id=run_id,
        trader_id=trader_id,
        historical_trades_path=str(trades_path),
        historical_pnl_path=str(pnl_path),
        weekly_signal_mask_path=str(weekly_mask_path),
        weekly_returns_path=str(weekly_returns_path),
        metadata={"mask_source": "real_backtest", "cutoff_date": str(dates[-1].date())},
    )
    store.replace_trader_weekly_returns(run_id=run_id, trader_id=trader_id, rows=weekly_returns.to_dict(orient="records"))
    store.replace_trader_weekly_signal_mask(run_id=run_id, trader_id=trader_id, rows=weekly_mask.to_dict(orient="records"))

    specs = {
        trader_id: PromotedTraderSpec(trader_id=trader_id, asset="AAPL", timeframe="D1", long_rules=["r1"], short_rules=[], origin_experiment_id="exp1"),
    }
    config = PPOPortfolioConfig(device_preference="cpu", artifact_root=_tmp_artifact_root(), min_history_weeks=20)
    builder = PortfolioDatasetBuilder(config, store)
    dataset = builder.build_dataset(
        promoted_specs=specs,
        history_loader=lambda _tid: pd.DataFrame({"date": dates, "equity": np.ones(len(dates)) * 10000.0}),
    )
    trader_idx = dataset.trader_index[trader_id]
    observed_mask = dataset.active_mask[:, trader_idx]
    assert observed_mask[:8].sum() == 0.0
    assert observed_mask[8:28].sum() == 20.0
    refresh_meta = dataset.trade_metadata["dataset_refresh"]
    assert refresh_meta["mask_source_by_trader"][trader_id] == "real_backtest"
    assert refresh_meta["cutoff_date_by_trader"][trader_id] == str(dates[-1].date())
    assert refresh_meta["mask_source"] == "real_backtest"


def test_dataset_builder_marks_fallback_proxy_when_refresh_artifacts_missing() -> None:
    store = StateStore(db_path=_tmp_db_path())
    dates = pd.date_range("2024-01-05", periods=60, freq="W-FRI")
    specs = {
        "tr_B": PromotedTraderSpec(trader_id="tr_B", asset="MSFT", timeframe="D1", long_rules=["r2"], short_rules=[], origin_experiment_id="exp2"),
    }
    history = pd.DataFrame(
        {
            "date": dates,
            "equity": 10000.0 * np.cumprod(1.0 + np.where(np.arange(len(dates)) % 2 == 0, 0.01, 0.0)),
        }
    )
    builder = PortfolioDatasetBuilder(PPOPortfolioConfig(device_preference="cpu", artifact_root=_tmp_artifact_root(), min_history_weeks=20), store)
    dataset = builder.build_dataset(promoted_specs=specs, history_loader=lambda _tid: history)
    refresh_meta = dataset.trade_metadata["dataset_refresh"]
    assert refresh_meta["mask_source_by_trader"]["tr_B"] == "fallback_proxy"
    assert refresh_meta["mask_source"] == "fallback_proxy"


def test_supervisor_monthly_refresh_updates_traceability(monkeypatch) -> None:
    db_path = _tmp_db_path()
    supervisor = DevelopmentOperationalSupervisor(db_path=db_path)
    spec = PromotedTraderSpec(trader_id="tr_A", asset="AAPL", timeframe="D1", long_rules=["r1"], short_rules=[], origin_experiment_id="exp1")

    monkeypatch.setattr(supervisor, "get_all_promoted_specs", lambda: {"tr_A": spec})
    monkeypatch.setattr(
        supervisor.portfolio_refresh_service,
        "refresh",
        lambda symbols: SimpleNamespace(
            cutoff_date="2026-04-30",
            refreshed_symbols=list(symbols),
            n_requested_symbols=len(list(symbols)),
            n_refreshed_symbols=len(list(symbols)),
            status="ok",
            metadata={},
        ),
    )

    def _fake_backtest(promoted_spec, *, refresh_reason="development_cycle"):
        supervisor._set_backtest_entry(
            str(promoted_spec.trader_id),
            {
                "status": "ready",
                "asset": promoted_spec.asset,
                "timeframe": promoted_spec.timeframe,
                "cutoff_date": "2026-04-30",
                "mask_source": "real_backtest",
                "updated_at": "2026-04-30T12:00:00+00:00",
            },
        )

    monkeypatch.setattr(supervisor, "_run_backtest_for_promoted", _fake_backtest)
    monkeypatch.setattr(supervisor.portfolio_manager_agent, "sync_universe", lambda promoted_specs: None)
    supervisor.portfolio_manager_agent._latest_dataset = SimpleNamespace(
        trade_metadata={"dataset_refresh": {"mask_source": "real_backtest", "cutoff_date": "2026-04-30"}}
    )
    monkeypatch.setattr(
        supervisor.portfolio_manager_agent,
        "run_monthly_refresh_and_fine_tune",
        lambda **kwargs: {"model_version": "ppo_test_v1", "checkpoint_path": "dummy.ckpt"},
    )

    result = supervisor.run_portfolio_monthly_refresh(force=True, as_of="2026-04-01")
    status = supervisor.get_status()
    assert result["status"] == "completed"
    assert result["cutoff_date"] == "2026-04-30"
    assert status["portfolio_last_refresh_cutoff_date"] == "2026-04-30"
    assert status["portfolio_last_refresh_mask_source"] == "real_backtest"
    assert status["portfolio_last_refresh_backtests_status"] == "ok"


def test_supervisor_force_manual_retraining_and_rebalance(monkeypatch) -> None:
    supervisor = DevelopmentOperationalSupervisor(db_path=_tmp_db_path())
    monkeypatch.setattr(
        supervisor,
        "run_portfolio_monthly_refresh",
        lambda **kwargs: {"status": "completed", "model_info": {"model_version": "ppo_manual_v1"}},
    )
    supervisor._runtime = SimpleNamespace(
        force_rebalance_now=lambda **kwargs: {
            "status": "manual_rebalance_executed",
            "selected_tickers": ["tr_A"],
            "weights": {"tr_A": 0.4},
            "target_cash_weight": 0.6,
        }
    )
    out = supervisor.force_portfolio_retraining_and_rebalance()
    status = supervisor.get_status()
    assert out["refresh"]["status"] == "completed"
    assert out["rebalance"]["status"] == "manual_rebalance_executed"
    assert status["portfolio_last_manual_retrain_at"] is not None
    assert status["portfolio_last_manual_rebalance_at"] is not None
    assert status["portfolio_last_manual_retrain_and_rebalance_at"] is not None


def test_supervisor_force_manual_retraining_only(monkeypatch) -> None:
    supervisor = DevelopmentOperationalSupervisor(db_path=_tmp_db_path())
    monkeypatch.setattr(
        supervisor,
        "run_portfolio_monthly_refresh",
        lambda **kwargs: {"status": "completed", "model_info": {"model_version": "ppo_manual_only_v1"}},
    )
    out = supervisor.force_portfolio_retraining_only()
    status = supervisor.get_status()
    assert out["refresh"]["status"] == "completed"
    assert out["rebalance"]["status"] == "not_requested"
    assert status["portfolio_last_manual_retrain_at"] is not None
    assert status["portfolio_last_manual_retrain_only_at"] is not None
