"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from enum import Enum
from pydantic import BaseModel, ConfigDict, field_validator
import pandas as pd
from typing import Dict, List

from pyeventbt.backtest.core.backtest_results import BacktestResults

class WalkforwardType(str, Enum):
    ANCHORED = 'ANCHORED'
    UNANCHORED = 'UNANCHORED'
    
    
class WalkForwardResults(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    backtest_results: BacktestResults
    retrainting_timestamps: List[pd.Timestamp]
    hyperparameters_track: pd.DataFrame
    
    @field_validator("retrainting_timestamps", mode="before")
    @classmethod
    def transform_timstamps(cls, raw: List[str]) -> List[pd.Timestamp]:
        return [pd.Timestamp(timestamp) for timestamp in raw]
    
    @field_validator("hyperparameters_track", mode="before")
    @classmethod
    def transform(cls, raw: pd.DataFrame | List[Dict[str, int | float]] ) -> pd.DataFrame:
        # check if the passed raw is type dataframe
        if isinstance(raw, pd.DataFrame):
            return raw
        # check if the passed raw is type list
        elif isinstance(raw, list):
            # transform the list to dataframe
            return pd.DataFrame(raw)
    
    def to_csv(self, path: str):
        """
        Save the backtest results, retrainting timestamps, and hyperparameters track to CSV files.

        Args:
            path (str): The path where the CSV files will be saved.
            
        >>> walk_forward_results.to_csv("path/to/save/directory")
        """
        self.backtest_results.save(path + "/backtest_results.csv")
        pd.DataFrame(self.retrainting_timestamps).to_csv(path + "/retrainting_timestamps.csv")
        pd.DataFrame(self.hyperparameters_track).to_csv(path + "/hyperparameters_track.csv")
    
    @staticmethod
    def from_csv(path: str):
        """
        Load walk forward results from CSV files.

        Parameters:
        - path (str): The path to the directory containing the CSV files.

        Returns:
        - WalkForwardResults: An instance of WalkForwardResults containing the loaded data.
        >>> walk_forward_results = WalkForwardResults.from_csv("path/to/load/directory")
        """
        backtest_results = BacktestResults.load(path + "/backtest_results.csv")
        retrainting_timestamps = list(map(lambda x: x[1], pd.read_csv(path + "/retrainting_timestamps.csv").values))
        hyperparameters_track = pd.read_csv(path + "/hyperparameters_track.csv")
        return WalkForwardResults(backtest_results=backtest_results, retrainting_timestamps=retrainting_timestamps, hyperparameters_track=hyperparameters_track)