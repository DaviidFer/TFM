from __future__ import annotations

from typing import Sequence

import pandas as pd

from indicators import build_feature_library, validate_feature_frame


def build_features(
    data_ohlc: pd.DataFrame,
    *,
    periods: Sequence[int] = tuple(range(2, 101, 2)),
    breakout_periods: Sequence[int] = (2, 5, 10, 20),
    seq_periods: Sequence[int] = (2, 3, 4),
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Wrapper del toolbox de indicadores para ejecutar fuera del notebook.

    La capa predictora queda cerrada a 11 familias de indicadores con
    periodización homogénea. Se unifican los periodos por defecto con el
    backtest para evitar reglas válidas offline que luego no existan en runtime.
    """
    features = build_feature_library(
        data_ohlc=data_ohlc[["open", "high", "low", "close"]],
        periods=list(periods),
        breakout_periods=list(breakout_periods),
        seq_periods=list(seq_periods),
        dropna=dropna,
    )
    validate_feature_frame(features, periods=periods)
    return features

