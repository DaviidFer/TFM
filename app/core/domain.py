from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class DomainScope:
    """
    Dominio oficial de la Fase 1.

    El objetivo es congelar alcance para evitar deriva de requisitos
    mientras se migra de notebook a aplicacion modular.
    """

    domain_name: str = "stocks_etfs_daily"
    data_root: Path = Path("datos/Stocks")
    timeframe: str = "D1"
    allowed_asset_classes: Sequence[str] = field(default_factory=lambda: ("Stock", "ETF"))
    sample_assets: Sequence[str] = field(default_factory=lambda: ("AAPL", "MSFT", "NVDA"))
    notebook_as_toolbox: bool = True
    notebook_runtime_allowed: bool = False


PHASE1_SCOPE = DomainScope()

