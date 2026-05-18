from __future__ import annotations

import pandas as pd
from types import SimpleNamespace

from app.contracts import PromotedTraderSpec
from app.runtime.development_operational_supervisor import DevelopmentOperationalSupervisor


def test_supervisor_hydrates_backtest_entry_from_store(tmp_path) -> None:
    db_path = tmp_path / "supervisor.sqlite"
    supervisor = DevelopmentOperationalSupervisor(db_path=db_path)

    trader_id = "tr_A"
    run_id = "bt_tr_A_20260508180000"
    spec = PromotedTraderSpec(
        trader_id=trader_id,
        asset="AAPL",
        timeframe="D1",
        long_rules=["rsi_lt_30"],
        short_rules=["rsi_gt_70"],
        origin_experiment_id="exp_A",
    )
    supervisor._promoted_registry[trader_id] = spec

    artifact_dir = tmp_path / "trader_backtests" / trader_id / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pnl_path = artifact_dir / "historical_pnl.csv"
    trades_path = artifact_dir / "historical_trades.csv"
    weekly_mask_path = artifact_dir / "weekly_mask.csv"
    weekly_returns_path = artifact_dir / "weekly_returns.csv"

    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "balance": [10000.0, 10120.0],
            "equity": [10000.0, 10110.0],
        }
    ).to_csv(pnl_path, index=False)
    pd.DataFrame(columns=["entry_time", "exit_time", "pnl"]).to_csv(trades_path, index=False)
    pd.DataFrame(columns=["week_end", "active"]).to_csv(weekly_mask_path, index=False)
    pd.DataFrame(columns=["week_end", "weekly_return"]).to_csv(weekly_returns_path, index=False)

    supervisor.ctx.store.upsert_trader_backtest_run(
        run_id=run_id,
        trader_id=trader_id,
        asset="AAPL",
        timeframe="D1",
        start_date="2024-01-01",
        end_date="2024-01-02",
        cutoff_date="2024-01-02",
        rules_hash="hash_rules",
        price_data_fingerprint="hash_prices",
        status="completed",
        summary={
            "n_trades": 1,
            "initial_capital": 10000.0,
            "final_balance": 10120.0,
            "final_equity": 10110.0,
            "trade_stats": {"total_trades": 1, "winning_trades": 1},
        },
    )
    supervisor.ctx.store.upsert_trader_backtest_artifacts(
        run_id=run_id,
        trader_id=trader_id,
        historical_trades_path=str(trades_path),
        historical_pnl_path=str(pnl_path),
        weekly_signal_mask_path=str(weekly_mask_path),
        weekly_returns_path=str(weekly_returns_path),
        metadata={"mask_source": "real_backtest", "cutoff_date": "2024-01-02"},
    )

    entry = supervisor.get_backtest_entry(trader_id)

    assert entry["status"] == "ready"
    assert entry["run_id"] == run_id
    assert entry["final_balance"] == 10120.0
    assert entry["final_equity"] == 10110.0
    assert entry["long_rules"] == ["rsi_lt_30"]
    assert entry["short_rules"] == ["rsi_gt_70"]
    assert len(entry["chart_rows"]) == 2
    assert str(entry["chart_rows"][0]["date"]).startswith("2024-01-01")


def test_start_operational_runtime_requires_minimum_traders(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "supervisor.sqlite"
    supervisor = DevelopmentOperationalSupervisor(db_path=db_path)
    supervisor._thread = SimpleNamespace(is_alive=lambda: True)
    supervisor._promoted_registry["tr_A"] = PromotedTraderSpec(
        trader_id="tr_A",
        asset="AAPL",
        timeframe="D1",
        long_rules=["rsi_lt_30"],
        short_rules=[],
        origin_experiment_id="exp_A",
    )

    def _unexpected_mt5_probe(*args, **kwargs):
        raise AssertionError("MT5 no debe conectarse por debajo del minimo de traders")

    monkeypatch.setattr(supervisor, "ensure_mt5_execution_ready", _unexpected_mt5_probe)

    out = supervisor.start_operational_runtime()

    assert out["started"] is False
    assert out["reason"] == "minimum_traders_not_reached"
    assert out["n_traders"] == 1
    assert out["min_traders_required"] == 5
    assert out["mt5_connected"] is False


def test_ensure_operational_runtime_is_deferred_while_development_is_active(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "supervisor.sqlite"
    supervisor = DevelopmentOperationalSupervisor(db_path=db_path)
    for idx in range(5):
        trader_id = f"tr_{idx}"
        supervisor._promoted_registry[trader_id] = PromotedTraderSpec(
            trader_id=trader_id,
            asset=f"SYM{idx}",
            timeframe="D1",
            long_rules=["r1"],
            short_rules=[],
            origin_experiment_id=f"exp_{idx}",
        )
    supervisor._develop_enabled.set()

    def _unexpected_mt5_probe(*args, **kwargs):
        raise AssertionError("MT5 no debe inicializarse mientras el desarrollo siga activo")

    monkeypatch.setattr(supervisor, "ensure_mt5_execution_ready", _unexpected_mt5_probe)

    started = supervisor._ensure_operational_runtime()

    assert started is False
    assert supervisor._runtime is None
    status = supervisor.get_status()
    assert status["operational_runtime_started"] is False


def test_operational_runtime_does_not_fallback_to_paper_when_mt5_is_unavailable(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "supervisor.sqlite"
    supervisor = DevelopmentOperationalSupervisor(db_path=db_path)
    supervisor._thread = SimpleNamespace(is_alive=lambda: True)
    for idx in range(5):
        trader_id = f"tr_{idx}"
        supervisor._promoted_registry[trader_id] = PromotedTraderSpec(
            trader_id=trader_id,
            asset=f"SYM{idx}",
            timeframe="D1",
            long_rules=["r1"],
            short_rules=[],
            origin_experiment_id=f"exp_{idx}",
        )

    monkeypatch.setattr(
        supervisor,
        "ensure_mt5_execution_ready",
        lambda **kwargs: {"connected": False, "reason": "El trading algorítmico está desactivado en MT5.", "mode": "live_mt5"},
    )

    out = supervisor.start_operational_runtime()

    assert out["started"] is False
    assert out["reason"] == "mt5_not_ready"
    assert out["mt5_connected"] is False
    assert "algorítmico" in str(out["mt5_reason"])
    assert supervisor._runtime is None
    status = supervisor.get_status()
    assert status["operational_runtime_started"] is False
    assert status["mt5_connected"] is False
    assert "algorítmico" in str(status["operational_runtime_last_error"])


def test_build_strategy_restores_default_dev_settings(tmp_path) -> None:
    supervisor = DevelopmentOperationalSupervisor(db_path=tmp_path / "supervisor.sqlite")

    strategy = supervisor._build_strategy("AAPL")

    assert strategy["split"]["lookback_years"] == 10
    assert 50 <= int(strategy["validation"]["monkey_is"]["n_monkeys"]) <= 100
    assert 50 <= int(strategy["validation"]["monkey_oos"]["n_monkeys"]) <= 100
