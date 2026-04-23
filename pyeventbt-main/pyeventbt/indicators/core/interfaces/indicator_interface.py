"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from abc import ABC, abstractmethod
import pandas as pd

from abc import ABC, abstractmethod
import pandas as pd

class IIndicator(ABC):
    """
    Interface for indicator classes.
    """

    @staticmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Abstract method to calculate the indicator values.

        Parameters:
            data (pd.DataFrame): The input data for the calculation (obtained from modules.DATA_PROVIDER.get_latest_bars(event.symbol, StrategyTimeframes.ONE_HOUR, ema_window + 1)).

        Returns:
            pd.Series: The calculated indicator values.
        """
        pass