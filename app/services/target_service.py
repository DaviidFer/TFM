from __future__ import annotations

from typing import Dict

import pandas as pd

from app.toolbox.definicion_target import run_target_para_bloques


def apply_target_to_blocks(blocks: Dict[str, pd.DataFrame], *, out_col: str = "Target") -> Dict[str, pd.DataFrame]:
    data, data_oos, data_2025, data_final = run_target_para_bloques(
        data=blocks["data_is"],
        data_oos=blocks["data_oos"],
        data_2025=blocks["data_2025"],
        data_final=blocks["data_final"],
        out_col=out_col,
        dropna_target=True,
    )
    out = dict(blocks)
    out["data_is"] = data
    out["data_oos"] = data_oos
    out["data_2025"] = data_2025
    out["data_final"] = data_final
    return out

