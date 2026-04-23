"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

#import numpy as np
import pandas as pd
#import os
import matplotlib.pyplot as plt
#from functools import lru_cache
#from typing import Callable
#from sklearn.linear_model import LinearRegression
#from pydantic import BaseModel
#from scipy.stats import norm
#from enum import Enum
#from pyeventbt.utils.utils import print_percentage_bar

# Silence Matplotlib Futurewarnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="matplotlib")

        
class BacktestResults:

    #backtest_results_save_name = 'backtest_results.csv'

    def __init__(self, backtest_pnl: pd.DataFrame, trades: pd.DataFrame) -> None:
        self._backtest_pnl = backtest_pnl
        self._pnl = backtest_pnl.astype(float)
        self._returns = backtest_pnl.EQUITY.pct_change()
        self._trades = trades

    @property
    def pnl(self):
        return self._pnl
    
    @property
    def returns(self):
        return self._returns
    
    @property
    def trades(self):
        return self._trades
    
    @property
    def backtest_pnl(self):
        return self._backtest_pnl        
    
    def plot(self):
        ax = self.pnl[['EQUITY', 'BALANCE']].plot(title='Backtest')
        ax.legend(['Equity', 'Balance'])
        ax.margins(x=0.01, y=0.01)
        plt.show()
    
    def plot_old(self):
        self.pnl[['EQUITY', 'BALANCE']].plot()
        plt.show()
    
