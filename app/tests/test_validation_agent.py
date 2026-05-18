from __future__ import annotations

from pathlib import Path

import pandas as pd

import app.agents.validation_agent as validation_module
from app.agents.base import AgentContext
from app.agents.developer_agent import DevelopmentOutput
from app.agents.validation_agent import ValidationAgent
from app.contracts import CandidateRules, EventType, ExperimentConfig
from app.storage import StateStore


def _dummy_dev_output() -> DevelopmentOutput:
    exp = ExperimentConfig(
        experiment_id="exp_test_validation",
        asset="AAPL",
        timeframe="D1",
        split_policy="default",
        model_families=["rulefit"],
        parameters={},
    )
    candidates = CandidateRules(
        experiment_id=exp.experiment_id,
        asset=exp.asset,
        long_rules=["long_1", "long_2"],
        short_rules=["short_1"],
    )
    empty = pd.DataFrame()
    return DevelopmentOutput(
        experiment_config=exp,
        candidate_rules=candidates,
        blocks={"data_is": empty, "data_oos": empty, "data_2025": empty},
        candidates_by_family={"rulefit": {"long": empty, "short": empty}},
    )


def test_validation_agent_does_not_promote_when_stability_is_empty(monkeypatch, tmp_path: Path) -> None:
    ctx = AgentContext(
        store=StateStore(db_path=tmp_path / "supervisor.sqlite"),
        artifacts_root=tmp_path,
    )
    agent = ValidationAgent(ctx)
    dev = _dummy_dev_output()

    monkeypatch.setattr(
        validation_module,
        "run_validation_pipeline",
        lambda **kwargs: {
            "stability": {"winners_long_stable": [], "winners_short_stable": []},
            "decor_long": pd.DataFrame({"regla": ["fallback_long"]}),
            "decor_short": pd.DataFrame({"regla": ["fallback_short"]}),
            "validation_profile": {"stability_selection": {"top_n_long": 15, "top_n_short": 15}},
        },
    )

    out = agent.validate_and_promote(dev, promote_if_empty=True)

    assert out.promoted_spec is None
    assert out.report.passed_long == 0
    assert out.report.passed_short == 0
    assert out.report.notes == "no_rules_passed_validation"

    events = list(ctx.store.list_events(limit=20))
    assert any(e["event_type"] == EventType.VALIDATION_COMPLETED.value for e in events)
    assert not any(e["event_type"] == EventType.TRADER_PROMOTED.value for e in events)
