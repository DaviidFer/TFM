"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from pyeventbt.strategy.core.modules import Modules
from ..core.interfaces.sizing_engine_interface import ISizingEngine
from ..core.configurations.sizing_engine_configurations import RiskPctSizingConfig
from pyeventbt.events.events import SignalEvent
from pyeventbt.data_provider.core.interfaces.data_provider_interface import IDataProvider
from pyeventbt.portfolio_handler.core.entities.suggested_order import SuggestedOrder
import pyeventbt.trading_context.trading_context as trading_context
from pyeventbt.utils.utils import Utils, check_platform_compatibility
from decimal import Decimal

    
class MT5RiskPctSizing(ISizingEngine):
    """MT5 implementation of risk percentage position sizing strategy.
    
    This sizing engine calculates position size based on a percentage of account equity
    that the trader is willing to risk on a single trade. The position size is calculated
    using the stop loss distance and the risk percentage to determine the appropriate volume.
    """
    
    def __init__(self, configs: RiskPctSizingConfig, trading_context: trading_context.TypeContext = trading_context.TypeContext.BACKTEST) -> None:
        """Initialize the risk percentage sizing engine.
        
        Args:
            configs: Configuration containing the risk percentage
            trading_context: Trading context (BACKTEST or LIVE), defaults to BACKTEST
        """
        if trading_context== "BACKTEST":
            from pyeventbt.broker.mt5_broker.mt5_simulator_wrapper import Mt5SimulatorWrapper as mt5
        else:
            check_platform_compatibility()
            try:
                import MetaTrader5 as mt5
            except ImportError:
                mt5 = None
        
        self.mt5 = mt5
        self.risk_pct = configs.risk_pct
    
    # This is done inside this class and not in a Utils file beacuse we need the data provider for getting info from the
    # latest tick, and the Utils class is ideally filled with staticmethods without any dependency. We'll need to reuse this
    # code for maybe risk manager to compute exposures or leverages
    def _convert_currency_amount_to_another_currency(self, amount: float, from_ccy: str, to_ccy: str, latest_tick: dict) -> float:
        """Convert currency amount from one currency to another using latest tick data.
        
        This method is implemented in this class rather than Utils because it requires
        the data provider for getting latest tick information, and Utils class is
        ideally filled with static methods without dependencies. This code may be
        reused for risk manager to compute exposures or leverages.
        
        Args:
            amount: Amount to convert
            from_ccy: Source currency code
            to_ccy: Target currency code  
            latest_tick: Latest tick data containing bid/ask prices
            
        Returns:
            float: Converted amount in target currency
        """
        all_fx_symbols = ("AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD", "CADCHF", "CADJPY", "CHFJPY", "EURAUD", "EURCAD",
                            "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD", "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD",
                            "GBPUSD", "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD", "USDCAD", "USDCHF", "USDJPY", "USDSEK", "USDNOK")
        
        # Convert the currencies to uppercase
        from_ccy = from_ccy.upper()
        to_ccy = to_ccy.upper()
        
        # If the currencies are the same, return the amount
        if from_ccy == to_ccy:
            return amount
        
        # Find the symbol in all_fx_symbols that contains both the margin_ccy and the account_currency
        # For example, if margin_ccy = "USD" and account_currency = "EUR", the symbol will be "EURUSD"
        fx_symbol = [s for s in all_fx_symbols if from_ccy in s and to_ccy in s][0]
        fx_symbol_base = fx_symbol[:3]

        # Get the conversion rate.
        last_price = latest_tick['bid']
        
        # Convert the amount to the new currency and return it
        converted_amount = amount / last_price if fx_symbol_base == to_ccy else amount * last_price
        return converted_amount
    
    
    def get_suggested_order(self, signal_event: SignalEvent, modules: Modules) -> SuggestedOrder:
        """Generate a suggested order with position size calculated using risk percentage.
        
        The position size is calculated based on:
        - Account equity and risk percentage
        - Stop loss distance from entry price
        - Symbol properties (tick size, contract size, etc.)
        - Currency conversion between account and profit currencies
        
        Args:
            signal_event: The trading signal event containing trade information
            modules: Trading modules containing data provider and other services
            
        Returns:
            SuggestedOrder: Order suggestion with calculated volume based on risk percentage
            
        Raises:
            Exception: If risk percentage is invalid (<= 0) or stop loss is not set
        """
        risk_pct = Decimal(str(self.risk_pct))
        
        # first check if the risk_pct is legal and if there is sl present
        if risk_pct <= 0:
            raise Exception(f"Risk percentage {risk_pct} is not valid.")
        if signal_event.sl == 0:
            raise Exception(f"Stop loss is 0. PCT_RISK sizing method requires a stop loss to be present in the signal event.")
        
        # Get the account and symbol info objects
        account_info = self.mt5.account_info()
        symbol_info  = self.mt5.symbol_info(signal_event.symbol)
        
        # Get the estimated entry price. If market order, use the current price, otherwise use the order price
        #entry_price = 0.0
        
        last_tick = modules.DATA_PROVIDER.get_latest_tick(signal_event.symbol)
        if signal_event.order_type == "MARKET":
            # Get the latest tick and use the ask price for buy and bid price for sell
            #last_tick = data_provider.get_latest_tick(signal_event.symbol)
            entry_price: Decimal = last_tick['ask'] if signal_event.signal_type == "BUY" else last_tick['bid']
        else:
            # Get the order price
            entry_price: Decimal = signal_event.order_price
        
        # Get the symbol properties needed to accurately calculate the position size
        equity                  = account_info.equity
        volume_step             = symbol_info.volume_step               # the minimum volume change step when placing an order
        tick_size               = symbol_info.trade_tick_size           # the smallest possible price change
        contract_size           = symbol_info.trade_contract_size       # the size of 1 lot
        account_currency        = account_info.currency                 # the account currency
        symbol_profit_currency  = symbol_info.currency_profit           # the profit currency of the symbol
        
        tick_value_profit_ccy   = contract_size * tick_size    # Amount gained or losed for a full contract and a tick price move
        tick_value_account_ccy  = Utils.convert_currency_amount_to_another_currency(tick_value_profit_ccy, symbol_profit_currency, account_currency, modules.DATA_PROVIDER)
        
        # Calculate the position size
        price_distance = int(abs(entry_price - signal_event.sl) / tick_size)    # compute price distance in integer price min sizes
        monetary_risk = equity * risk_pct / 100                                 # compute monetary risk in account currency
        volume = monetary_risk / (price_distance * tick_value_account_ccy)      # compute volume in lots
        volume = round(volume / volume_step) * volume_step                      # normalize to volume_step units

        return SuggestedOrder(signal_event=signal_event,
                            volume=volume)