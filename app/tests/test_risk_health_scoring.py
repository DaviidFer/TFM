from __future__ import annotations

from app.contracts import DesignRiskProfile, RiskAction, RiskLimitsConfig, TraderForwardMetrics
from app.services.risk import evaluate_trader_health


def _design_profile() -> DesignRiskProfile:
    return DesignRiskProfile(
        trader_id="tr_A",
        asset="AAPL",
        timeframe="D1",
        promoted_at="2026-01-01T00:00:00+00:00",
        sharpe_design=1.5,
        profit_factor_design=1.9,
        max_drawdown_design=0.10,
        avg_loss_design=-10.0,
        avg_win_design=18.0,
        winrate_design=0.58,
        expectancy_design=4.5,
        max_losing_streak_design=3,
        trades_design=60,
    )


def test_trader_sano_returns_keep() -> None:
    snapshot = evaluate_trader_health(
        _design_profile(),
        TraderForwardMetrics(
            trader_id="tr_A",
            asset="AAPL",
            timeframe="D1",
            evaluation_run_id="risk_1",
            promoted_at="2026-01-01T00:00:00+00:00",
            evaluation_date="2026-03-01T00:00:00+00:00",
            forward_start="2026-01-01",
            forward_end="2026-03-01",
            shadow_trades=18,
            shadow_sharpe=1.2,
            shadow_profit_factor=1.6,
            shadow_max_drawdown=0.11,
            shadow_avg_loss=-10.5,
            shadow_avg_win=16.0,
            shadow_winrate=0.54,
            shadow_expectancy=2.5,
            shadow_losing_streak=3,
            signal_count=18,
            ppo_selected_count=12,
        ),
        current_state="live",
        limits=RiskLimitsConfig(),
    )
    assert snapshot.action == RiskAction.KEEP.value
    assert snapshot.health_score >= 60.0


def test_pocos_trades_no_retraining() -> None:
    """Sin evidencia forward suficiente el trader se mantiene LIVE (KEEP)."""
    snapshot = evaluate_trader_health(
        _design_profile(),
        TraderForwardMetrics(
            trader_id="tr_A",
            asset="AAPL",
            timeframe="D1",
            evaluation_run_id="risk_2",
            promoted_at="2026-01-01T00:00:00+00:00",
            evaluation_date="2026-01-20T00:00:00+00:00",
            forward_start="2026-01-01",
            forward_end="2026-01-20",
            shadow_trades=4,
            shadow_sharpe=-1.0,
            shadow_profit_factor=0.4,
            shadow_max_drawdown=0.25,
            insufficient_evidence=True,
        ),
        current_state="live",
        limits=RiskLimitsConfig(),
    )
    assert snapshot.action == RiskAction.KEEP.value
    assert snapshot.retrain_request is None


def test_profit_factor_deteriorado_returns_retraining() -> None:
    """Con evidencia suficiente y deterioro significativo -> RETRAINING."""
    snapshot = evaluate_trader_health(
        _design_profile(),
        TraderForwardMetrics(
            trader_id="tr_A",
            asset="AAPL",
            timeframe="D1",
            evaluation_run_id="risk_3",
            promoted_at="2026-01-01T00:00:00+00:00",
            evaluation_date="2026-03-01T00:00:00+00:00",
            forward_start="2026-01-01",
            forward_end="2026-03-01",
            shadow_trades=16,
            shadow_sharpe=0.8,
            shadow_profit_factor=1.1,
            shadow_max_drawdown=0.12,
            shadow_avg_loss=-11.0,
            shadow_winrate=0.45,
            shadow_expectancy=0.5,
        ),
        current_state="live",
        limits=RiskLimitsConfig(),
    )
    assert snapshot.action == RiskAction.RETRAINING.value
    assert snapshot.retrain_request is not None


def test_drawdown_severo_returns_retraining() -> None:
    snapshot = evaluate_trader_health(
        _design_profile(),
        TraderForwardMetrics(
            trader_id="tr_A",
            asset="AAPL",
            timeframe="D1",
            evaluation_run_id="risk_4",
            promoted_at="2026-01-01T00:00:00+00:00",
            evaluation_date="2026-04-01T00:00:00+00:00",
            forward_start="2026-01-01",
            forward_end="2026-04-01",
            shadow_trades=30,
            shadow_sharpe=0.3,
            shadow_profit_factor=0.8,
            shadow_max_drawdown=0.19,
            shadow_avg_loss=-18.0,
            shadow_losing_streak=6,
            shadow_expectancy=-2.0,
            signal_count=30,
            ppo_selected_count=5,
            ppo_blocked_count=4,
        ),
        current_state="live",
        limits=RiskLimitsConfig(),
    )
    assert snapshot.action == RiskAction.RETRAINING.value
    assert snapshot.retrain_request is not None


def test_many_trades_and_low_health_returns_retraining() -> None:
    snapshot = evaluate_trader_health(
        _design_profile(),
        TraderForwardMetrics(
            trader_id="tr_A",
            asset="AAPL",
            timeframe="D1",
            evaluation_run_id="risk_5",
            promoted_at="2026-01-01T00:00:00+00:00",
            evaluation_date="2026-05-01T00:00:00+00:00",
            forward_start="2026-01-01",
            forward_end="2026-05-01",
            shadow_trades=40,
            shadow_sharpe=-0.4,
            shadow_profit_factor=0.2,
            shadow_max_drawdown=0.30,
            shadow_avg_loss=-30.0,
            shadow_winrate=0.20,
            shadow_expectancy=-8.0,
            shadow_losing_streak=10,
            signal_count=40,
            ppo_selected_count=0,
            ppo_blocked_count=12,
        ),
        current_state="live",
        limits=RiskLimitsConfig(),
    )
    assert snapshot.action == RiskAction.RETRAINING.value
    assert snapshot.retrain_request is not None
