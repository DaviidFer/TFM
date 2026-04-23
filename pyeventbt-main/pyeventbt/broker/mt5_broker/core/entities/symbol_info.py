"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from pydantic import BaseModel
from decimal import Decimal

class SymbolInfo(BaseModel):
    custom: bool
    chart_mode: int
    select: bool
    visible: bool
    session_deals: int
    session_buy_orders: int
    session_sell_orders: int
    volume: Decimal
    volumehigh: Decimal
    volumelow: Decimal
    time: int
    digits: int
    spread: int
    spread_float: bool
    ticks_bookdepth: int
    trade_calc_mode: int
    trade_mode: int
    start_time: int
    expiration_time: int
    trade_stops_level: int
    trade_freeze_level: int
    trade_exemode: int
    swap_mode: int
    swap_rollover3days: int
    margin_hedged_use_leg: bool
    expiration_mode: int
    filling_mode: int
    order_mode: int
    order_gtc_mode: int
    option_mode: int
    option_right: int
    bid: Decimal
    bidhigh: Decimal
    bidlow: Decimal
    ask: Decimal
    askhigh: Decimal
    asklow: Decimal
    last: Decimal
    lasthigh: Decimal
    lastlow: Decimal
    volume_real: Decimal
    volumehigh_real: Decimal
    volumelow_real: Decimal
    option_strike: Decimal
    point: Decimal
    trade_tick_value: Decimal
    trade_tick_value_profit: Decimal
    trade_tick_value_loss: Decimal
    trade_tick_size: Decimal
    trade_contract_size: Decimal
    trade_accrued_interest: Decimal
    trade_face_value: Decimal
    trade_liquidity_rate: Decimal
    volume_min: Decimal
    volume_max: Decimal
    volume_step: Decimal
    volume_limit: Decimal
    swap_long: Decimal
    swap_short: Decimal
    margin_initial: Decimal
    margin_maintenance: Decimal
    session_volume: Decimal
    session_turnover: Decimal
    session_interest: Decimal
    session_buy_orders_volume: Decimal
    session_sell_orders_volume: Decimal
    session_open: Decimal
    session_close: Decimal
    session_aw: Decimal
    session_price_settlement: Decimal
    session_price_limit_min: Decimal
    session_price_limit_max: Decimal
    margin_hedged: Decimal
    price_change: Decimal
    price_volatility: Decimal
    price_theoretical: Decimal
    price_greeks_delta: Decimal
    price_greeks_theta: Decimal
    price_greeks_gamma: Decimal
    price_greeks_vega: Decimal
    price_greeks_rho: Decimal
    price_greeks_omega: Decimal
    price_sensitivity: Decimal
    basis: str
    category: str
    currency_base: str
    currency_profit: str
    currency_margin: str
    bank: str
    description: str
    exchange: str
    formula: str
    isin: str
    name: str
    page: str
    path: str
