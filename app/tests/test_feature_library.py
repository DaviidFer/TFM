from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.feature_service import build_features
from app.toolbox.indicators import (
    ACTIVE_INDICATOR_FAMILIES,
    NON_PREDICTIVE_COLUMNS,
    validate_feature_frame,
)


def test_feature_library_uses_only_closed_indicator_set() -> None:
    rows = 420
    idx = pd.date_range("2020-01-01", periods=rows, freq="D")
    base = np.linspace(100.0, 150.0, rows)
    wave = np.sin(np.linspace(0.0, 12.0, rows))

    close = pd.Series(base + wave, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0]) + 0.2
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.5
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.5
    ohlc = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)

    periods = (4, 8, 14, 22, 30)
    features = build_features(ohlc, periods=periods, dropna=False)
    report = validate_feature_frame(features, periods=periods)

    predictive_cols = [c for c in features.columns if c not in NON_PREDICTIVE_COLUMNS]

    assert list(features.columns[:4]) == list(NON_PREDICTIVE_COLUMNS)
    assert len(predictive_cols) == len(ACTIVE_INDICATOR_FAMILIES) * len(periods)
    assert not report["duplicates"]
    assert not report["invalid_predictive"]
    assert not report["missing_expected"]
    assert not report["unexpected_predictive"]
    assert not report["all_nan_after_warmup"]
