from __future__ import annotations

from typing import Dict

import pandas as pd

from particion_IS_OOS import run_particion_is_oos


def split_is_oos_holdout(
    df_features: pd.DataFrame,
    *,
    is_pct: float = 0.5,
    oos_pct: float = 0.5,
    holdout_year: int = 2025,
    holdout_enabled: bool = True,
    lookback_years: int = 10,
) -> Dict[str, pd.DataFrame]:
    data, data_oos, data_2025, data_main, data_final = run_particion_is_oos(
        df_full=df_features,
        is_pct=is_pct,
        oos_pct=oos_pct,
        holdout_year=holdout_year,
        holdout_enabled=holdout_enabled,
        lookback_years=lookback_years,
    )
    return {
        "data_is": data,
        "data_oos": data_oos,
        "data_2025": data_2025,
        "data_main": data_main,
        "data_final": data_final,
    }

