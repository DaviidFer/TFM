"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from typing import Protocol
from pyeventbt.events.events import FillEvent

class ITradeArchiver(Protocol):

    def archive_trade(self, fill_event: FillEvent) -> None:
        raise NotImplementedError()
    
    def get_trade_archive(self) -> dict[int, FillEvent]:
        raise NotImplementedError()
    
    def export_csv_trade_archive(self, file_path: str) -> None:
        raise NotImplementedError()