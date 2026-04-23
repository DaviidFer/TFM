"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from typing import Protocol

class IPlatform(Protocol):

    def initialize(self):
        raise NotImplementedError()
    
    def login(self):
        raise NotImplementedError()
    
    def shutdown(self):
        raise NotImplementedError()
    
    def version(self):
        raise NotImplementedError()
    
    def last_error(self):
        raise NotImplementedError()


class IAccountInfo(Protocol):

    def account_info(self):
        raise NotImplementedError()


class ITerminalInfo(Protocol):
    
    def terminal_info(self):
        raise NotImplementedError()


class ISymbol(Protocol):
    
    def symbols_total(self):
        raise NotImplementedError()
    
    def symbols_get(self):
        raise NotImplementedError()
    
    def symbol_info(self):
        raise NotImplementedError()
    
    def symbol_info_tick(self):
        raise NotImplementedError()
    
    def symbol_select(self):
        raise NotImplementedError()


class IMarketBook(Protocol):
        
    def market_book_add(self):
        raise NotImplementedError()
    
    def market_book_get(self):
        raise NotImplementedError()
    
    def market_book_release(self):
        raise NotImplementedError()


class IMarketData(Protocol):
    
    def copy_rates_from(self):
        raise NotImplementedError()
    
    def copy_rates_from_pos(self):
        raise NotImplementedError()
    
    def copy_rates_range(self):
        raise NotImplementedError()
    
    def copy_ticks_from(self):
        raise NotImplementedError()
    
    def copy_ticks_range(self):
        raise NotImplementedError()


class IOrder(Protocol):

    def orders_total(self):
        raise NotImplementedError()
    
    def orders_get(self):
        raise NotImplementedError()
    
    def order_calc_margin(self):
        raise NotImplementedError()
    
    def order_calc_profit(self):
        raise NotImplementedError()
    
    def order_check(self):
        raise NotImplementedError()
    
    def order_send(self):
        raise NotImplementedError()


class IPosition(Protocol):

    def positions_total(self):
        raise NotImplementedError()
    
    def positions_get(self):
        raise NotImplementedError()


class IHistory(Protocol):

    def history_orders_total(self):
        raise NotImplementedError()
    
    def history_orders_get(self):
        raise NotImplementedError()
    
    def history_deals_total(self):
        raise NotImplementedError()
    
    def history_deals_get(self):
        raise NotImplementedError()