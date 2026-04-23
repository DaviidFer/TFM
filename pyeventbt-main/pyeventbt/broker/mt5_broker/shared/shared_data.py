"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from pyeventbt.broker.mt5_broker.core.entities.init_credentials import InitCredentials
from pyeventbt.broker.mt5_broker.core.entities.account_info import AccountInfo
from pyeventbt.broker.mt5_broker.core.entities.terminal_info import TerminalInfo
from pyeventbt.broker.mt5_broker.core.entities.symbol_info import SymbolInfo
from os import path
from decimal import Decimal
import yaml

class SharedData():

    # This gets executed before __init__
    last_error_code: tuple

    # Create the needed data objects
    credentials: InitCredentials = None
    terminal_info: TerminalInfo = None
    account_info: AccountInfo = None
    symbol_info: dict[str, SymbolInfo] = None

    def __init__(self):
        # Initial values
        #print("\n+-------------- KOMALOGIC EVENT-DRIVEN BACKTEST INITIATED --------------+")
        #print("| - Initializing MT5 Simulator Shared Data...")
        #SharedData.connected = False
        SharedData.last_error_code = (-1, 'generic fail')
        
        # Load default data
        self._load_default_terminal_info()
        self._load_default_account_info()
        self._load_default_symbols_info()
    
    @staticmethod
    def decimal_constructor(loader, node):
        # Custom constructor for converting YAML floats to Decimal
        value = loader.construct_scalar(node)
        return Decimal(value)
    
    def _load_yaml_file(self, filepath:str):
        try:
            # Register the custom constructor for this load operation
            yaml.add_constructor('tag:yaml.org,2002:float', self.decimal_constructor, yaml.SafeLoader)

            with open(filepath, 'r') as file:
                # Use yaml.load with SafeLoader to utilize the custom constructor
                yaml_data = yaml.load(file, Loader=yaml.SafeLoader)
                #yaml_data = yaml.safe_load(file)
            return yaml_data
        
        except Exception as e:
            print(f"Error loading yaml file {filepath}: {e}")
            return False

    def _load_default_account_info(self) -> None:
        #print("| - Loading and preparing Default Account Info...")
        account_info_file_path = path.join(path.dirname(__file__), "default_account_info.yaml")
        yaml_data = self._load_yaml_file(account_info_file_path)
        SharedData.account_info = AccountInfo(**yaml_data)
    
    def _load_default_terminal_info(self) -> None:
        #print("| - Loading and preparing Default Terminal Info...")
        terminal_info_file_path = path.join(path.dirname(__file__), "default_terminal_info.yaml")
        yaml_data = self._load_yaml_file(terminal_info_file_path)
        SharedData.terminal_info = TerminalInfo(**yaml_data)

    def _load_default_symbols_info(self) -> None:
        #print("| - Loading and preparing Default Symbols Info...")
        symbols_info_file_path = path.join(path.dirname(__file__), "default_symbols_info.yaml")
        yaml_data = self._load_yaml_file(symbols_info_file_path)

        # yaml_data is a dict(str, dict). We have to convert the inner dicts to SymbolInfo objects
        for symbol in yaml_data:
            yaml_data[symbol] = SymbolInfo(**yaml_data[symbol])

        SharedData.symbol_info = yaml_data

