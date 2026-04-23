"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from pandas import DataFrame
from pandas.core.api import Series as Series
from .core.interfaces.indicator_interface import IIndicator
import numpy as np
from numba import njit


class KAMA(IIndicator):
    """Kaufman Adaptive Moving Average (KAMA) indicator."""
    
    @staticmethod
    @njit
    def __compute_kama(close: np.ndarray, n_period: int, sc_fastest: float, sc_slowest: float) -> np.ndarray:
        """
        Compute KAMA values using Numba for performance.
        """
        n = len(close)
        kama = np.empty(n, dtype=np.float64)
        kama[:] = np.nan  # Initialize with NaN
        
        for i in range(n_period - 1, n):
            # Calculate Efficiency Ratio
            momentum = abs(close[i] - close[i - n_period])
            volatility = 0.0
            for j in range(i - n_period + 1, i + 1):
                volatility += abs(close[j] - close[j - 1])
            
            if volatility == 0:
                er = 0.0
            else:
                er = momentum / volatility
            
            # Calculate Smoothing Constant
            sc = (er * (sc_fastest - sc_slowest) + sc_slowest) ** 2
            
            # Calculate KAMA
            if i == n_period - 1:
                kama[i] = close[i]
            else:
                kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
        
        return kama
    
    @staticmethod
    def compute(close: np.ndarray, n_period: int = 10, period_fast: int = 2, period_slow: int = 30) -> np.ndarray:
        """Calculate the KAMA indicator values.
        
        Parameters:
            close (np.ndarray): Close prices as a numpy array.
            n_period (int): The number of periods to look back from the current time. Sliding window. Default is 10.
            period_fast (int): The fast period parameter to calculate the fast period smoothing constant. Default is 2.
            period_slow (int): The slow period parameter to calculate the slow period smoothing constant, should be greater than the fast parameter. Default is 30.
        
        Returns:
            np.ndarray: The calculated KAMA indicator values as a numpy array.
            
        Usage:
        ```python
        close_prices = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        kama_values = KAMA.compute(close_prices, n_period=5)
        ```
        """
        if len(close) < n_period:
            raise ValueError(f"Close array length must be at least {n_period}")
        
        # Smoothing Constants
        sc_fastest = 2 / (period_fast + 1)
        sc_slowest = 2 / (period_slow + 1)
        
        return KAMA._KAMA__compute_kama(close, n_period, sc_fastest, sc_slowest)
    

class ATR(IIndicator):
    """
    Average True Range (ATR) indicator.
    """
    @staticmethod
    @njit
    def __compute_tr(high, low, close):       
        """
        Compute the True Range (TR) values.
        """
        n = len(high)
        tr = np.empty(n, dtype=np.float64)
        tr[0] = high[0] - low[0]  # Initialize TR[0] with a valid TR value
        for i in range(1, n):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i - 1])
            tr3 = abs(low[i] - close[i - 1])
            tr[i] = max(tr1, tr2, tr3)
        return tr
    
    @staticmethod
    @njit
    def __compute_atr_sma(tr, period):
        """
        Compute ATR using Simple Moving Average (SMA) method.
        """
        n = len(tr)
        atr = np.empty(n, dtype=np.float64)
        atr[:] = np.nan  # Initialize with NaN
        rolling_sum = 0.0

        for i in range(n):
            rolling_sum += tr[i]
            if i >= period:
                rolling_sum -= tr[i - period]
            if i >= period - 1:
                atr[i] = rolling_sum / period
        return atr
    
    @staticmethod
    @njit
    def __compute_atr_ema(tr, period):
        """
        Compute ATR using Exponential Moving Average (EMA) method.
        """
        n = len(tr)
        atr = np.empty(n, dtype=np.float64)
        atr[:] = np.nan  # Initialize with NaN
        multiplier = 2.0 / (period + 1)

        # Initialize EMA with the SMA of the first 'period' TR values
        ema = 0.0
        for i in range(period):
            ema += tr[i]
        ema /= period

        atr[period - 1] = ema  # Set the first ATR value

        for i in range(period, n):
            ema = (tr[i] - ema) * multiplier + ema
            atr[i] = ema

        return atr

    @staticmethod
    def compute(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, method: str='sma') -> np.ndarray:
        """
        Compute the Average True Range (ATR) using SMA or EMA with high performance.
        
        Parameters:
            high (np.ndarray): High prices as a numpy array.
            low (np.ndarray): Low prices as a numpy array.
            close (np.ndarray): Close prices as a numpy array.
            period (int): The period for ATR calculation.
            method (str): "sma" for Simple Moving Average or "ema" for Exponential Moving Average.
            
        Returns:
            np.ndarray: ATR values as a numpy array.
        """
        n = len(high)
        if (len(low) != n) or (len(close) != n):
            raise ValueError("High, low, and close arrays must have the same length.")

        if (method != "sma") and (method != "ema"):
            raise ValueError("Method must be either 'sma' or 'ema'.")

        # Compute True Range - using mangled name as I defined methods as private to avoid user confusion
        tr = ATR._ATR__compute_tr(high, low, close)

        if method == "sma":
            return ATR._ATR__compute_atr_sma(tr, period)
        else:  # method == "ema"
            return ATR._ATR__compute_atr_ema(tr, period)


class SMA(IIndicator):
    """Simple Moving Average (SMA) indicator."""
    
    @staticmethod
    @njit
    def __compute_sma(close: np.ndarray, period: int) -> np.ndarray:
        """
        Compute SMA values using Numba for performance.
        """
        n = len(close)
        sma = np.empty(n, dtype=np.float64)
        sma[:] = np.nan  # Initialize with NaN
        rolling_sum = 0.0

        for i in range(n):
            rolling_sum += close[i]
            if i >= period:
                rolling_sum -= close[i - period]
            if i >= period - 1:
                sma[i] = rolling_sum / period
        return sma
    
    @staticmethod
    def compute(close: np.ndarray, period: int) -> np.ndarray:
        """Calculate the Simple Moving Average (SMA) indicator values.
        
        Parameters:
            close (np.ndarray): Close prices as a numpy array.
            period (int): The number of periods for the moving average.
        
        Returns:
            np.ndarray: The calculated SMA indicator values as a numpy array.
            
        Usage:
        ```python
        close_prices = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        sma_values = SMA.compute(close_prices, period=5)
        ```
        """
        if len(close) < period:
            raise ValueError(f"Close array length must be at least {period}")
        
        return SMA._SMA__compute_sma(close, period)


class EMA(IIndicator):
    """Exponential Moving Average (EMA) indicator."""
    
    @staticmethod
    @njit
    def __compute_ema(close: np.ndarray, period: int) -> np.ndarray:
        """
        Compute EMA values using Numba for performance.
        """
        n = len(close)
        ema = np.empty(n, dtype=np.float64)
        ema[:] = np.nan  # Initialize with NaN
        multiplier = 2.0 / (period + 1)

        # Initialize EMA with the SMA of the first 'period' values
        sma_sum = 0.0
        for i in range(period):
            sma_sum += close[i]
        ema_value = sma_sum / period

        ema[period - 1] = ema_value  # Set the first EMA value

        for i in range(period, n):
            ema_value = (close[i] - ema_value) * multiplier + ema_value
            ema[i] = ema_value

        return ema
    
    @staticmethod
    def compute(close: np.ndarray, period: int) -> np.ndarray:
        """Calculate the Exponential Moving Average (EMA) indicator values.
        
        Parameters:
            close (np.ndarray): Close prices as a numpy array.
            period (int): The number of periods for the moving average.
        
        Returns:
            np.ndarray: The calculated EMA indicator values as a numpy array.
            
        Usage:
        ```python
        close_prices = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        ema_values = EMA.compute(close_prices, period=5)
        ```
        """
        if len(close) < period:
            raise ValueError(f"Close array length must be at least {period}")
        
        return EMA._EMA__compute_ema(close, period)


class RSI(IIndicator):
    """Relative Strength Index (RSI) indicator."""
    
    @staticmethod
    @njit
    def __compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
        """
        Compute RSI values using Numba for performance.
        """
        n = len(close)
        rsi = np.empty(n, dtype=np.float64)
        rsi[:] = np.nan
        
        if n < period + 1:
            return rsi
        
        # Calculate price changes
        gains = 0.0
        losses = 0.0
        
        # Initial average gain and loss
        for i in range(1, period + 1):
            change = close[i] - close[i - 1]
            if change > 0:
                gains += change
            else:
                losses += abs(change)
        
        avg_gain = gains / period
        avg_loss = losses / period
        
        if avg_loss == 0:
            rsi[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[period] = 100.0 - (100.0 / (1.0 + rs))
        
        # Calculate RSI for remaining periods using smoothed averages
        for i in range(period + 1, n):
            change = close[i] - close[i - 1]
            gain = change if change > 0 else 0.0
            loss = abs(change) if change < 0 else 0.0
            
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
            
            if avg_loss == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    @staticmethod
    def compute(close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate the Relative Strength Index (RSI) indicator values.
        
        Parameters:
            close (np.ndarray): Close prices as a numpy array.
            period (int): The number of periods for RSI calculation. Default is 14.
        
        Returns:
            np.ndarray: The calculated RSI indicator values as a numpy array.
            
        Usage:
        ```python
        close_prices = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        rsi_values = RSI.compute(close_prices, period=14)
        ```
        """
        if len(close) < period + 1:
            raise ValueError(f"Close array length must be at least {period + 1}")
        
        return RSI._RSI__compute_rsi(close, period)


class ADX(IIndicator):
    """Average Directional Index (ADX) indicator."""
    
    @staticmethod
    @njit
    def __compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> tuple:
        """
        Compute ADX, +DI, and -DI values using Numba for performance.
        Returns (adx, plus_di, minus_di)
        """
        n = len(close)
        adx = np.empty(n, dtype=np.float64)
        plus_di = np.empty(n, dtype=np.float64)
        minus_di = np.empty(n, dtype=np.float64)
        adx[:] = np.nan
        plus_di[:] = np.nan
        minus_di[:] = np.nan
        
        # Calculate True Range and Directional Movement
        tr = np.empty(n, dtype=np.float64)
        plus_dm = np.empty(n, dtype=np.float64)
        minus_dm = np.empty(n, dtype=np.float64)
        
        tr[0] = high[0] - low[0]
        plus_dm[0] = 0.0
        minus_dm[0] = 0.0
        
        for i in range(1, n):
            # True Range
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i - 1])
            tr3 = abs(low[i] - close[i - 1])
            tr[i] = max(tr1, tr2, tr3)
            
            # Directional Movement
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            else:
                plus_dm[i] = 0.0
            
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            else:
                minus_dm[i] = 0.0
        
        # Smooth TR and DM
        atr = 0.0
        smoothed_plus_dm = 0.0
        smoothed_minus_dm = 0.0
        
        # Initial smoothed values
        for i in range(period):
            atr += tr[i]
            smoothed_plus_dm += plus_dm[i]
            smoothed_minus_dm += minus_dm[i]
        
        # Calculate DI
        if atr != 0:
            plus_di[period - 1] = 100.0 * smoothed_plus_dm / atr
            minus_di[period - 1] = 100.0 * smoothed_minus_dm / atr
        
        # Subsequent smoothed values
        for i in range(period, n):
            atr = atr - atr / period + tr[i]
            smoothed_plus_dm = smoothed_plus_dm - smoothed_plus_dm / period + plus_dm[i]
            smoothed_minus_dm = smoothed_minus_dm - smoothed_minus_dm / period + minus_dm[i]
            
            if atr != 0:
                plus_di[i] = 100.0 * smoothed_plus_dm / atr
                minus_di[i] = 100.0 * smoothed_minus_dm / atr
        
        # Calculate DX and ADX
        dx = np.empty(n, dtype=np.float64)
        dx[:] = np.nan
        
        for i in range(period - 1, n):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum != 0:
                dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
        
        # Smooth DX to get ADX
        adx_sum = 0.0
        count = 0
        for i in range(period - 1, n):
            if not np.isnan(dx[i]):
                adx_sum += dx[i]
                count += 1
                if count == period:
                    adx[i] = adx_sum / period
                    break
        
        start_idx = period - 1 + period - 1
        if start_idx < n:
            for i in range(start_idx + 1, n):
                if not np.isnan(dx[i]):
                    adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def compute(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> tuple:
        """Calculate the Average Directional Index (ADX) indicator values.
        
        Parameters:
            high (np.ndarray): High prices as a numpy array.
            low (np.ndarray): Low prices as a numpy array.
            close (np.ndarray): Close prices as a numpy array.
            period (int): The number of periods for ADX calculation. Default is 14.
        
        Returns:
            tuple: (adx, plus_di, minus_di) - ADX, +DI, and -DI values as numpy arrays.
            
        Usage:
        ```python
        high_prices = np.array([102, 104, 103, 105, 107])
        low_prices = np.array([100, 101, 100, 102, 104])
        close_prices = np.array([101, 103, 102, 104, 106])
        adx, plus_di, minus_di = ADX.compute(high_prices, low_prices, close_prices, period=14)
        ```
        """
        n = len(high)
        if len(low) != n or len(close) != n:
            raise ValueError("High, low, and close arrays must have the same length.")
        if n < period * 2:
            raise ValueError(f"Array length must be at least {period * 2}")
        
        return ADX._ADX__compute_adx(high, low, close, period)


class Momentum(IIndicator):
    """Momentum indicator."""
    
    @staticmethod
    @njit
    def __compute_momentum(close: np.ndarray, period: int) -> np.ndarray:
        """
        Compute Momentum values using Numba for performance.
        """
        n = len(close)
        momentum = np.empty(n, dtype=np.float64)
        momentum[:] = np.nan
        
        for i in range(period, n):
            momentum[i] = close[i] - close[i - period]
        
        return momentum
    
    @staticmethod
    def compute(close: np.ndarray, period: int = 10) -> np.ndarray:
        """Calculate the Momentum indicator values.
        
        Parameters:
            close (np.ndarray): Close prices as a numpy array.
            period (int): The number of periods for momentum calculation. Default is 10.
        
        Returns:
            np.ndarray: The calculated Momentum indicator values as a numpy array.
            
        Usage:
        ```python
        close_prices = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        momentum_values = Momentum.compute(close_prices, period=5)
        ```
        """
        if len(close) < period + 1:
            raise ValueError(f"Close array length must be at least {period + 1}")
        
        return Momentum._Momentum__compute_momentum(close, period)


class BollingerBands(IIndicator):
    """Bollinger Bands indicator."""
    
    @staticmethod
    @njit
    def __compute_bollinger(close: np.ndarray, period: int, std_dev: float) -> tuple:
        """
        Compute Bollinger Bands using Numba for performance.
        Returns (upper_band, middle_band, lower_band)
        """
        n = len(close)
        upper = np.empty(n, dtype=np.float64)
        middle = np.empty(n, dtype=np.float64)
        lower = np.empty(n, dtype=np.float64)
        upper[:] = np.nan
        middle[:] = np.nan
        lower[:] = np.nan
        
        for i in range(period - 1, n):
            # Calculate SMA
            sma = 0.0
            for j in range(i - period + 1, i + 1):
                sma += close[j]
            sma /= period
            middle[i] = sma
            
            # Calculate standard deviation
            variance = 0.0
            for j in range(i - period + 1, i + 1):
                diff = close[j] - sma
                variance += diff * diff
            std = np.sqrt(variance / period)
            
            # Calculate bands
            upper[i] = sma + (std_dev * std)
            lower[i] = sma - (std_dev * std)
        
        return upper, middle, lower
    
    @staticmethod
    def compute(close: np.ndarray, period: int = 20, std_dev: float = 2.0) -> tuple:
        """Calculate the Bollinger Bands indicator values.
        
        Parameters:
            close (np.ndarray): Close prices as a numpy array.
            period (int): The number of periods for the moving average. Default is 20.
            std_dev (float): Number of standard deviations for the bands. Default is 2.0.
        
        Returns:
            tuple: (upper_band, middle_band, lower_band) as numpy arrays.
            
        Usage:
        ```python
        close_prices = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        upper, middle, lower = BollingerBands.compute(close_prices, period=20, std_dev=2.0)
        ```
        """
        if len(close) < period:
            raise ValueError(f"Close array length must be at least {period}")
        
        return BollingerBands._BollingerBands__compute_bollinger(close, period, std_dev)


class DonchianChannels(IIndicator):
    """Donchian Channels indicator."""
    
    @staticmethod
    @njit
    def __compute_donchian(high: np.ndarray, low: np.ndarray, period: int) -> tuple:
        """
        Compute Donchian Channels using Numba for performance.
        Returns (upper_channel, middle_channel, lower_channel)
        """
        n = len(high)
        upper = np.empty(n, dtype=np.float64)
        middle = np.empty(n, dtype=np.float64)
        lower = np.empty(n, dtype=np.float64)
        upper[:] = np.nan
        middle[:] = np.nan
        lower[:] = np.nan
        
        for i in range(period - 1, n):
            # Find highest high and lowest low
            highest = high[i - period + 1]
            lowest = low[i - period + 1]
            
            for j in range(i - period + 2, i + 1):
                if high[j] > highest:
                    highest = high[j]
                if low[j] < lowest:
                    lowest = low[j]
            
            upper[i] = highest
            lower[i] = lowest
            middle[i] = (highest + lowest) / 2.0
        
        return upper, middle, lower
    
    @staticmethod
    def compute(high: np.ndarray, low: np.ndarray, period: int = 20) -> tuple:
        """Calculate the Donchian Channels indicator values.
        
        Parameters:
            high (np.ndarray): High prices as a numpy array.
            low (np.ndarray): Low prices as a numpy array.
            period (int): The number of periods for the channels. Default is 20.
        
        Returns:
            tuple: (upper_channel, middle_channel, lower_channel) as numpy arrays.
            
        Usage:
        ```python
        high_prices = np.array([102, 104, 103, 105, 107])
        low_prices = np.array([100, 101, 100, 102, 104])
        upper, middle, lower = DonchianChannels.compute(high_prices, low_prices, period=20)
        ```
        """
        n = len(high)
        if len(low) != n:
            raise ValueError("High and low arrays must have the same length.")
        if n < period:
            raise ValueError(f"Array length must be at least {period}")
        
        return DonchianChannels._DonchianChannels__compute_donchian(high, low, period)


class MACD(IIndicator):
    """Moving Average Convergence Divergence (MACD) indicator."""
    
    @staticmethod
    @njit
    def __compute_ema_for_macd(close: np.ndarray, period: int) -> np.ndarray:
        """
        Compute EMA values for MACD calculation.
        """
        n = len(close)
        ema = np.empty(n, dtype=np.float64)
        ema[:] = np.nan
        multiplier = 2.0 / (period + 1)
        
        # Initialize with SMA
        sma_sum = 0.0
        for i in range(period):
            sma_sum += close[i]
        ema_value = sma_sum / period
        ema[period - 1] = ema_value
        
        for i in range(period, n):
            ema_value = (close[i] - ema_value) * multiplier + ema_value
            ema[i] = ema_value
        
        return ema
    
    @staticmethod
    @njit
    def __compute_macd(close: np.ndarray, fast_period: int, slow_period: int, signal_period: int) -> tuple:
        """
        Compute MACD values using Numba for performance.
        Returns (macd_line, signal_line, histogram)
        """
        n = len(close)
        macd_line = np.empty(n, dtype=np.float64)
        signal_line = np.empty(n, dtype=np.float64)
        histogram = np.empty(n, dtype=np.float64)
        macd_line[:] = np.nan
        signal_line[:] = np.nan
        histogram[:] = np.nan
        
        # Calculate fast and slow EMAs
        fast_ema = MACD._MACD__compute_ema_for_macd(close, fast_period)
        slow_ema = MACD._MACD__compute_ema_for_macd(close, slow_period)
        
        # Calculate MACD line
        for i in range(slow_period - 1, n):
            macd_line[i] = fast_ema[i] - slow_ema[i]
        
        # Calculate signal line (EMA of MACD line)
        macd_values = macd_line[slow_period - 1:]
        multiplier = 2.0 / (signal_period + 1)
        
        # Initialize signal line with SMA
        sma_sum = 0.0
        count = 0
        for i in range(len(macd_values)):
            if not np.isnan(macd_values[i]):
                sma_sum += macd_values[i]
                count += 1
                if count == signal_period:
                    signal_value = sma_sum / signal_period
                    signal_line[slow_period - 1 + i] = signal_value
                    
                    # Calculate remaining signal values
                    for j in range(i + 1, len(macd_values)):
                        if not np.isnan(macd_values[j]):
                            signal_value = (macd_values[j] - signal_value) * multiplier + signal_value
                            signal_line[slow_period - 1 + j] = signal_value
                    break
        
        # Calculate histogram
        for i in range(n):
            if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i]):
                histogram[i] = macd_line[i] - signal_line[i]
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def compute(close: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
        """Calculate the MACD indicator values.
        
        Parameters:
            close (np.ndarray): Close prices as a numpy array.
            fast_period (int): The fast EMA period. Default is 12.
            slow_period (int): The slow EMA period. Default is 26.
            signal_period (int): The signal line EMA period. Default is 9.
        
        Returns:
            tuple: (macd_line, signal_line, histogram) as numpy arrays.
            
        Usage:
        ```python
        close_prices = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        macd_line, signal_line, histogram = MACD.compute(close_prices)
        ```
        """
        if len(close) < slow_period + signal_period:
            raise ValueError(f"Close array length must be at least {slow_period + signal_period}")
        
        return MACD._MACD__compute_macd(close, fast_period, slow_period, signal_period)


class KeltnerChannel(IIndicator):
    """Keltner Channel indicator."""
    
    @staticmethod
    @njit
    def __compute_keltner(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                          period: int, atr_period: int, multiplier: float) -> tuple:
        """
        Compute Keltner Channel using Numba for performance.
        Returns (upper_channel, middle_channel, lower_channel)
        """
        n = len(close)
        upper = np.empty(n, dtype=np.float64)
        middle = np.empty(n, dtype=np.float64)
        lower = np.empty(n, dtype=np.float64)
        upper[:] = np.nan
        middle[:] = np.nan
        lower[:] = np.nan
        
        # Calculate EMA of close for middle line
        ema_multiplier = 2.0 / (period + 1)
        sma_sum = 0.0
        for i in range(period):
            sma_sum += close[i]
        ema_value = sma_sum / period
        middle[period - 1] = ema_value
        
        for i in range(period, n):
            ema_value = (close[i] - ema_value) * ema_multiplier + ema_value
            middle[i] = ema_value
        
        # Calculate ATR
        tr = np.empty(n, dtype=np.float64)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i - 1])
            tr3 = abs(low[i] - close[i - 1])
            tr[i] = max(tr1, tr2, tr3)
        
        # EMA of ATR
        atr_ema_multiplier = 2.0 / (atr_period + 1)
        atr_sma_sum = 0.0
        for i in range(atr_period):
            atr_sma_sum += tr[i]
        atr_ema_value = atr_sma_sum / atr_period
        
        atr = np.empty(n, dtype=np.float64)
        atr[:] = np.nan
        atr[atr_period - 1] = atr_ema_value
        
        for i in range(atr_period, n):
            atr_ema_value = (tr[i] - atr_ema_value) * atr_ema_multiplier + atr_ema_value
            atr[i] = atr_ema_value
        
        # Calculate upper and lower bands
        start_idx = max(period - 1, atr_period - 1)
        for i in range(start_idx, n):
            if not np.isnan(middle[i]) and not np.isnan(atr[i]):
                upper[i] = middle[i] + multiplier * atr[i]
                lower[i] = middle[i] - multiplier * atr[i]
        
        return upper, middle, lower
    
    @staticmethod
    def compute(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                period: int = 20, atr_period: int = 10, multiplier: float = 2.0) -> tuple:
        """Calculate the Keltner Channel indicator values.
        
        Parameters:
            high (np.ndarray): High prices as a numpy array.
            low (np.ndarray): Low prices as a numpy array.
            close (np.ndarray): Close prices as a numpy array.
            period (int): The EMA period for the middle line. Default is 20.
            atr_period (int): The ATR period. Default is 10.
            multiplier (float): ATR multiplier for the bands. Default is 2.0.
        
        Returns:
            tuple: (upper_channel, middle_channel, lower_channel) as numpy arrays.
            
        Usage:
        ```python
        high_prices = np.array([102, 104, 103, 105, 107])
        low_prices = np.array([100, 101, 100, 102, 104])
        close_prices = np.array([101, 103, 102, 104, 106])
        upper, middle, lower = KeltnerChannel.compute(high_prices, low_prices, close_prices)
        ```
        """
        n = len(high)
        if len(low) != n or len(close) != n:
            raise ValueError("High, low, and close arrays must have the same length.")
        min_length = max(period, atr_period)
        if n < min_length:
            raise ValueError(f"Array length must be at least {min_length}")
        
        return KeltnerChannel._KeltnerChannel__compute_keltner(high, low, close, period, atr_period, multiplier)


class ADR(IIndicator):
    """Average Daily Range (ADR) indicator."""
    
    @staticmethod
    @njit
    def __compute_adr(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
        """
        Compute ADR values using Numba for performance.
        """
        n = len(high)
        adr = np.empty(n, dtype=np.float64)
        adr[:] = np.nan
        
        for i in range(period - 1, n):
            range_sum = 0.0
            for j in range(i - period + 1, i + 1):
                range_sum += high[j] - low[j]
            adr[i] = range_sum / period
        
        return adr
    
    @staticmethod
    def compute(high: np.ndarray, low: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate the Average Daily Range (ADR) indicator values.
        
        Parameters:
            high (np.ndarray): High prices as a numpy array.
            low (np.ndarray): Low prices as a numpy array.
            period (int): The number of periods for ADR calculation. Default is 14.
        
        Returns:
            np.ndarray: The calculated ADR indicator values as a numpy array.
            
        Usage:
        ```python
        high_prices = np.array([102, 104, 103, 105, 107])
        low_prices = np.array([100, 101, 100, 102, 104])
        adr_values = ADR.compute(high_prices, low_prices, period=14)
        ```
        """
        n = len(high)
        if len(low) != n:
            raise ValueError("High and low arrays must have the same length.")
        if n < period:
            raise ValueError(f"Array length must be at least {period}")
        
        return ADR._ADR__compute_adr(high, low, period)


class VWAP(IIndicator):
    """Volume Weighted Average Price (VWAP) indicator."""
    
    @staticmethod
    @njit
    def __compute_vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                       volume: np.ndarray, period: int = 0) -> np.ndarray:
        """
        Compute VWAP values using Numba for performance.
        If period is 0, calculates cumulative VWAP from the start.
        If period > 0, calculates rolling VWAP over the specified period.
        """
        n = len(close)
        vwap = np.empty(n, dtype=np.float64)
        vwap[:] = np.nan
        
        # Typical price
        typical_price = (high + low + close) / 3.0
        
        if period == 0:
            # Cumulative VWAP
            cumulative_tp_volume = 0.0
            cumulative_volume = 0.0
            
            for i in range(n):
                cumulative_tp_volume += typical_price[i] * volume[i]
                cumulative_volume += volume[i]
                
                if cumulative_volume > 0:
                    vwap[i] = cumulative_tp_volume / cumulative_volume
        else:
            # Rolling VWAP
            for i in range(period - 1, n):
                tp_volume_sum = 0.0
                volume_sum = 0.0
                
                for j in range(i - period + 1, i + 1):
                    tp_volume_sum += typical_price[j] * volume[j]
                    volume_sum += volume[j]
                
                if volume_sum > 0:
                    vwap[i] = tp_volume_sum / volume_sum
        
        return vwap
    
    @staticmethod
    def compute(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                volume: np.ndarray, period: int = 0) -> np.ndarray:
        """Calculate the Volume Weighted Average Price (VWAP) indicator values.
        
        Parameters:
            high (np.ndarray): High prices as a numpy array.
            low (np.ndarray): Low prices as a numpy array.
            close (np.ndarray): Close prices as a numpy array.
            volume (np.ndarray): Volume as a numpy array.
            period (int): The rolling period. If 0, calculates cumulative VWAP. Default is 0.
        
        Returns:
            np.ndarray: The calculated VWAP indicator values as a numpy array.
            
        Usage:
        ```python
        high_prices = np.array([102, 104, 103, 105, 107])
        low_prices = np.array([100, 101, 100, 102, 104])
        close_prices = np.array([101, 103, 102, 104, 106])
        volumes = np.array([1000, 1200, 900, 1100, 1300])
        vwap_values = VWAP.compute(high_prices, low_prices, close_prices, volumes)
        ```
        """
        n = len(high)
        if len(low) != n or len(close) != n or len(volume) != n:
            raise ValueError("High, low, close, and volume arrays must have the same length.")
        if period > 0 and n < period:
            raise ValueError(f"Array length must be at least {period}")
        
        return VWAP._VWAP__compute_vwap(high, low, close, volume, period)


class Stochastic(IIndicator):
    """Stochastic Oscillator indicator."""
    
    @staticmethod
    @njit
    def __compute_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                             k_period: int, d_period: int, smooth_k: int) -> tuple:
        """
        Compute Stochastic Oscillator using Numba for performance.
        Returns (%K, %D)
        """
        n = len(close)
        fast_k = np.empty(n, dtype=np.float64)
        slow_k = np.empty(n, dtype=np.float64)
        slow_d = np.empty(n, dtype=np.float64)
        fast_k[:] = np.nan
        slow_k[:] = np.nan
        slow_d[:] = np.nan
        
        # Calculate Fast %K
        for i in range(k_period - 1, n):
            # Find highest high and lowest low in the period
            highest = high[i - k_period + 1]
            lowest = low[i - k_period + 1]
            
            for j in range(i - k_period + 2, i + 1):
                if high[j] > highest:
                    highest = high[j]
                if low[j] < lowest:
                    lowest = low[j]
            
            # Calculate %K
            if highest - lowest != 0:
                fast_k[i] = 100.0 * (close[i] - lowest) / (highest - lowest)
            else:
                fast_k[i] = 50.0  # Neutral value when no range
        
        # Calculate Slow %K (SMA of Fast %K)
        for i in range(k_period - 1 + smooth_k - 1, n):
            k_sum = 0.0
            for j in range(i - smooth_k + 1, i + 1):
                if not np.isnan(fast_k[j]):
                    k_sum += fast_k[j]
            slow_k[i] = k_sum / smooth_k
        
        # Calculate Slow %D (SMA of Slow %K)
        start_idx = k_period - 1 + smooth_k - 1 + d_period - 1
        for i in range(start_idx, n):
            d_sum = 0.0
            count = 0
            for j in range(i - d_period + 1, i + 1):
                if not np.isnan(slow_k[j]):
                    d_sum += slow_k[j]
                    count += 1
            if count > 0:
                slow_d[i] = d_sum / count
        
        return slow_k, slow_d
    
    @staticmethod
    def compute(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                k_period: int = 14, d_period: int = 3, smooth_k: int = 3) -> tuple:
        """Calculate the Stochastic Oscillator indicator values.
        
        Parameters:
            high (np.ndarray): High prices as a numpy array.
            low (np.ndarray): Low prices as a numpy array.
            close (np.ndarray): Close prices as a numpy array.
            k_period (int): The lookback period for %K calculation. Default is 14.
            d_period (int): The period for %D smoothing. Default is 3.
            smooth_k (int): The smoothing period for %K. Default is 3.
        
        Returns:
            tuple: (%K, %D) as numpy arrays.
            
        Usage:
        ```python
        high_prices = np.array([102, 104, 103, 105, 107])
        low_prices = np.array([100, 101, 100, 102, 104])
        close_prices = np.array([101, 103, 102, 104, 106])
        k_values, d_values = Stochastic.compute(high_prices, low_prices, close_prices)
        ```
        """
        n = len(high)
        if len(low) != n or len(close) != n:
            raise ValueError("High, low, and close arrays must have the same length.")
        min_length = k_period + smooth_k + d_period
        if n < min_length:
            raise ValueError(f"Array length must be at least {min_length}")
        
        return Stochastic._Stochastic__compute_stochastic(high, low, close, k_period, d_period, smooth_k)


class CCI(IIndicator):
    """Commodity Channel Index (CCI) indicator."""
    
    @staticmethod
    @njit
    def __compute_cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """
        Compute CCI values using Numba for performance.
        """
        n = len(close)
        cci = np.empty(n, dtype=np.float64)
        cci[:] = np.nan
        
        # Typical Price
        typical_price = (high + low + close) / 3.0
        
        for i in range(period - 1, n):
            # Calculate SMA of typical price
            tp_sum = 0.0
            for j in range(i - period + 1, i + 1):
                tp_sum += typical_price[j]
            sma_tp = tp_sum / period
            
            # Calculate Mean Deviation
            mad = 0.0
            for j in range(i - period + 1, i + 1):
                mad += abs(typical_price[j] - sma_tp)
            mad /= period
            
            # Calculate CCI
            if mad != 0:
                cci[i] = (typical_price[i] - sma_tp) / (0.015 * mad)
            else:
                cci[i] = 0.0
        
        return cci
    
    @staticmethod
    def compute(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> np.ndarray:
        """Calculate the Commodity Channel Index (CCI) indicator values.
        
        Parameters:
            high (np.ndarray): High prices as a numpy array.
            low (np.ndarray): Low prices as a numpy array.
            close (np.ndarray): Close prices as a numpy array.
            period (int): The number of periods for CCI calculation. Default is 20.
        
        Returns:
            np.ndarray: The calculated CCI indicator values as a numpy array.
            
        Usage:
        ```python
        high_prices = np.array([102, 104, 103, 105, 107])
        low_prices = np.array([100, 101, 100, 102, 104])
        close_prices = np.array([101, 103, 102, 104, 106])
        cci_values = CCI.compute(high_prices, low_prices, close_prices, period=20)
        ```
        """
        n = len(high)
        if len(low) != n or len(close) != n:
            raise ValueError("High, low, and close arrays must have the same length.")
        if n < period:
            raise ValueError(f"Array length must be at least {period}")
        
        return CCI._CCI__compute_cci(high, low, close, period)


class WilliamsR(IIndicator):
    """Williams %R indicator."""
    
    @staticmethod
    @njit
    def __compute_williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """
        Compute Williams %R values using Numba for performance.
        """
        n = len(close)
        williams_r = np.empty(n, dtype=np.float64)
        williams_r[:] = np.nan
        
        for i in range(period - 1, n):
            # Find highest high and lowest low in the period
            highest = high[i - period + 1]
            lowest = low[i - period + 1]
            
            for j in range(i - period + 2, i + 1):
                if high[j] > highest:
                    highest = high[j]
                if low[j] < lowest:
                    lowest = low[j]
            
            # Calculate Williams %R
            if highest - lowest != 0:
                williams_r[i] = -100.0 * (highest - close[i]) / (highest - lowest)
            else:
                williams_r[i] = -50.0  # Neutral value when no range
        
        return williams_r
    
    @staticmethod
    def compute(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate the Williams %R indicator values.
        
        Parameters:
            high (np.ndarray): High prices as a numpy array.
            low (np.ndarray): Low prices as a numpy array.
            close (np.ndarray): Close prices as a numpy array.
            period (int): The lookback period for Williams %R calculation. Default is 14.
        
        Returns:
            np.ndarray: The calculated Williams %R indicator values as a numpy array (range -100 to 0).
            
        Usage:
        ```python
        high_prices = np.array([102, 104, 103, 105, 107])
        low_prices = np.array([100, 101, 100, 102, 104])
        close_prices = np.array([101, 103, 102, 104, 106])
        williams_r_values = WilliamsR.compute(high_prices, low_prices, close_prices, period=14)
        ```
        """
        n = len(high)
        if len(low) != n or len(close) != n:
            raise ValueError("High, low, and close arrays must have the same length.")
        if n < period:
            raise ValueError(f"Array length must be at least {period}")
        
        return WilliamsR._WilliamsR__compute_williams_r(high, low, close, period)


class ROC(IIndicator):
    """Rate of Change (ROC) indicator."""
    
    @staticmethod
    @njit
    def __compute_roc(close: np.ndarray, period: int) -> np.ndarray:
        """
        Compute ROC values using Numba for performance.
        """
        n = len(close)
        roc = np.empty(n, dtype=np.float64)
        roc[:] = np.nan
        
        for i in range(period, n):
            if close[i - period] != 0:
                roc[i] = 100.0 * (close[i] - close[i - period]) / close[i - period]
            else:
                roc[i] = 0.0
        
        return roc
    
    @staticmethod
    def compute(close: np.ndarray, period: int = 12) -> np.ndarray:
        """Calculate the Rate of Change (ROC) indicator values.
        
        Parameters:
            close (np.ndarray): Close prices as a numpy array.
            period (int): The number of periods for ROC calculation. Default is 12.
        
        Returns:
            np.ndarray: The calculated ROC indicator values as a numpy array (percentage).
            
        Usage:
        ```python
        close_prices = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        roc_values = ROC.compute(close_prices, period=12)
        ```
        """
        if len(close) < period + 1:
            raise ValueError(f"Close array length must be at least {period + 1}")
        
        return ROC._ROC__compute_roc(close, period)


class Aroon(IIndicator):
    """Aroon indicator (Aroon Up and Aroon Down)."""
    
    @staticmethod
    @njit
    def __compute_aroon(high: np.ndarray, low: np.ndarray, period: int) -> tuple:
        """
        Compute Aroon Up and Aroon Down values using Numba for performance.
        Returns (aroon_up, aroon_down)
        """
        n = len(high)
        aroon_up = np.empty(n, dtype=np.float64)
        aroon_down = np.empty(n, dtype=np.float64)
        aroon_up[:] = np.nan
        aroon_down[:] = np.nan
        
        for i in range(period, n):
            # Find periods since highest high
            highest_idx = i - period
            for j in range(i - period + 1, i + 1):
                if high[j] >= high[highest_idx]:
                    highest_idx = j
            
            # Find periods since lowest low
            lowest_idx = i - period
            for j in range(i - period + 1, i + 1):
                if low[j] <= low[lowest_idx]:
                    lowest_idx = j
            
            # Calculate Aroon Up and Down
            periods_since_high = i - highest_idx
            periods_since_low = i - lowest_idx
            
            aroon_up[i] = 100.0 * (period - periods_since_high) / period
            aroon_down[i] = 100.0 * (period - periods_since_low) / period
        
        return aroon_up, aroon_down
    
    @staticmethod
    def compute(high: np.ndarray, low: np.ndarray, period: int = 25) -> tuple:
        """Calculate the Aroon indicator values.
        
        Parameters:
            high (np.ndarray): High prices as a numpy array.
            low (np.ndarray): Low prices as a numpy array.
            period (int): The lookback period for Aroon calculation. Default is 25.
        
        Returns:
            tuple: (aroon_up, aroon_down) as numpy arrays (range 0 to 100).
            
        Usage:
        ```python
        high_prices = np.array([102, 104, 103, 105, 107])
        low_prices = np.array([100, 101, 100, 102, 104])
        aroon_up, aroon_down = Aroon.compute(high_prices, low_prices, period=25)
        ```
        """
        n = len(high)
        if len(low) != n:
            raise ValueError("High and low arrays must have the same length.")
        if n < period + 1:
            raise ValueError(f"Array length must be at least {period + 1}")
        
        return Aroon._Aroon__compute_aroon(high, low, period)