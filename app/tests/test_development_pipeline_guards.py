from __future__ import annotations

import pandas as pd

import app.agents.developer_agent as developer_module
from app.toolbox.ML_tools import genetico as genetico_module
from app.toolbox.ML_tools import rulefit as rulefit_module


def test_collect_rules_stops_when_iteration_adds_no_new_rules(monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_generate_candidate_rules(*args, **kwargs):
        calls["count"] += 1
        return {
            "rulefit": {
                "long": pd.DataFrame({"regla": ["long_rule"]}),
                "short": pd.DataFrame({"regla": ["short_rule"]}),
            }
        }

    monkeypatch.setattr(developer_module, "generate_candidate_rules", _fake_generate_candidate_rules)

    selected_long, selected_short, _, iteration_summaries, _, stop_reason = developer_module._collect_rules(
        blocks={"data_is": pd.DataFrame({"x": [1, 2, 3]})},
        families=("rulefit",),
        base_params={"rulefit": {}},
        target_long=3,
        target_short=3,
        max_iterations=12,
    )

    assert calls["count"] == 2
    assert selected_long == ["long_rule"]
    assert selected_short == ["short_rule"]
    assert len(iteration_summaries) == 2
    assert iteration_summaries[-1]["new_unique_long"] == 0
    assert iteration_summaries[-1]["new_unique_short"] == 0
    assert stop_reason == "no_new_rules"


def test_rulefit_multiseed_stops_after_stalled_seed_limit(monkeypatch) -> None:
    calls = {"count": 0}

    def _empty_seed(*args, **kwargs):
        calls["count"] += 1
        return pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(rulefit_module, "_build_rulefit_rules_single_seed", _empty_seed)

    long_df, short_df = rulefit_module.build_rulefit_rules_multiseed(
        data=pd.DataFrame({"Target": [1, 0, 1]}),
        target_n_rules=4,
        progress_every=0,
        max_seed_attempts=5,
        max_stalled_seeds=2,
    )

    assert calls["count"] == 2
    assert long_df.empty
    assert short_df.empty


def test_genetico_multiseed_stops_after_stalled_seed_limit(monkeypatch) -> None:
    calls = {"count": 0}

    def _empty_seed(*args, **kwargs):
        calls["count"] += 1
        return pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(genetico_module, "_build_genetic_rules_sqx_single_seed", _empty_seed)

    long_df, short_df = genetico_module.build_genetic_rules_sqx_multiseed(
        data=pd.DataFrame({"Target": [1, 0, 1]}),
        target_n_rules=4,
        progress_every=0,
        max_seed_attempts=5,
        max_stalled_seeds=2,
    )

    assert calls["count"] == 2
    assert long_df.empty
    assert short_df.empty
