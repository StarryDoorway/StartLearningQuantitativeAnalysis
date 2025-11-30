"""
Technical indicators module for the quantitative trading framework.

This module provides a comprehensive collection of technical indicators
commonly used in quantitative trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional


def sma(data: Union[pd.Series, np.ndarray], period: int) -> Union[pd.Series, np.ndarray]:
    """
    Simple Moving Average (SMA).
    
    Args:
        data: Price data
        period: Period for the moving average
        
    Returns:
        SMA values
    """
    if isinstance(data, pd.Series):
        return data.rolling(window=period).mean()
    else:
        return np.convolve(data, np.ones(period) / period, mode='valid')


def ema(data: Union[pd.Series, np.ndarray], period: int) -> Union[pd.Series, np.ndarray]:
    """
    Exponential Moving Average (EMA).
    
    Args:
        data: Price data
        period: Period for the moving average
        
    Returns:
        EMA values
    """
    if isinstance(data, pd.Series):
        return data.ewm(span=period).mean()
    else:
        alpha = 2 / (period + 1)
        ema_values = np.zeros_like(data)
        ema_values[0] = data[0]
        
        for i in range(1, len(data)):
            ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i-1]
        
        return ema_values


def rsi(data: Union[pd.Series, np.ndarray], period: int = 14) -> Union[pd.Series, np.ndarray]:
    """
    Relative Strength Index (RSI).
    
    Args:
        data: Price data
        period: Period for RSI calculation
        
    Returns:
        RSI values
    """
    if isinstance(data, pd.Series):
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    else:
        delta = np.diff(data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(data)
        avg_loss = np.zeros_like(data)
        
        # Initialize with simple average for the first period
        avg_gain[period-1] = np.mean(gain[:period])
        avg_loss[period-1] = np.mean(loss[:period])
        
        # Calculate subsequent values using EMA
        alpha = 1 / period
        for i in range(period, len(data)):
            avg_gain[i] = alpha * gain[i-1] + (1 - alpha) * avg_gain[i-1]
            avg_loss[i] = alpha * loss[i-1] + (1 - alpha) * avg_loss[i-1]
        
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
        return 100 - (100 / (1 + rs))


def macd(data: Union[pd.Series, np.ndarray], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
    """
    Moving Average Convergence Divergence (MACD).
    
    Args:
        data: Price data
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    ema_fast = ema(data, fast)
    ema_slow = ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def bollinger_bands(data: Union[pd.Series, np.ndarray], period: int = 20, std_dev: float = 2.0) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
    """
    Bollinger Bands.
    
    Args:
        data: Price data
        period: Period for moving average
        std_dev: Number of standard deviations
        
    Returns:
        Tuple of (Upper band, Middle band, Lower band)
    """
    if isinstance(data, pd.Series):
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    else:
        middle = np.zeros_like(data)
        upper = np.zeros_like(data)
        lower = np.zeros_like(data)
        
        for i in range(period-1, len(data)):
            window = data[i-period+1:i+1]
            middle[i] = np.mean(window)
            std = np.std(window)
            upper[i] = middle[i] + (std * std_dev)
            lower[i] = middle[i] - (std * std_dev)
        
        return upper, middle, lower


def stochastic(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], close: Union[pd.Series, np.ndarray], k_period: int = 14, d_period: int = 3) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
    """
    Stochastic Oscillator.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: %K period
        d_period: %D period
        
    Returns:
        Tuple of (%K, %D)
    """
    if isinstance(high, pd.Series) and isinstance(low, pd.Series) and isinstance(close, pd.Series):
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    else:
        k_percent = np.zeros_like(close)
        d_percent = np.zeros_like(close)
        
        for i in range(k_period-1, len(close)):
            window_low = low[i-k_period+1:i+1]
            window_high = high[i-k_period+1:i+1]
            
            lowest_low = np.min(window_low)
            highest_high = np.max(window_high)
            
            if highest_high != lowest_low:
                k_percent[i] = 100 * ((close[i] - lowest_low) / (highest_high - lowest_low))
            else:
                k_percent[i] = 50  # Default to 50 when no range
        
        # Calculate %D as a moving average of %K
        for i in range(d_period-1, len(k_percent)):
            d_percent[i] = np.mean(k_percent[i-d_period+1:i+1])
        
        return k_percent, d_percent


def atr(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], close: Union[pd.Series, np.ndarray], period: int = 14) -> Union[pd.Series, np.ndarray]:
    """
    Average True Range (ATR).
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Period for ATR calculation
        
    Returns:
        ATR values
    """
    if isinstance(high, pd.Series) and isinstance(low, pd.Series) and isinstance(close, pd.Series):
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    else:
        atr_values = np.zeros_like(close)
        
        # Calculate True Range
        true_range = np.zeros_like(close)
        for i in range(1, len(close)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            true_range[i] = max(tr1, tr2, tr3)
        
        # Calculate ATR using Wilder's smoothing
        atr_values[period-1] = np.mean(true_range[1:period])
        for i in range(period, len(close)):
            atr_values[i] = (atr_values[i-1] * (period - 1) + true_range[i]) / period
        
        return atr_values


def adx(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], close: Union[pd.Series, np.ndarray], period: int = 14) -> Union[pd.Series, np.ndarray]:
    """
    Average Directional Index (ADX).
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Period for ADX calculation
        
    Returns:
        ADX values
    """
    if isinstance(high, pd.Series) and isinstance(low, pd.Series) and isinstance(close, pd.Series):
        # Calculate True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Convert to pandas Series
        plus_dm = pd.Series(plus_dm, index=close.index)
        minus_dm = pd.Series(minus_dm, index=close.index)
        
        # Calculate Smoothed values
        atr_smoothed = true_range.ewm(com=period-1, min_periods=period).mean()
        plus_dm_smoothed = plus_dm.ewm(com=period-1, min_periods=period).mean()
        minus_dm_smoothed = minus_dm.ewm(com=period-1, min_periods=period).mean()
        
        # Calculate DI
        plus_di = 100 * (plus_dm_smoothed / atr_smoothed)
        minus_di = 100 * (minus_dm_smoothed / atr_smoothed)
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(com=period-1, min_periods=period).mean()
        
        return adx
    else:
        # For numpy arrays, we'll implement a simplified version
        # Calculate True Range
        true_range = np.zeros_like(close)
        for i in range(1, len(close)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            true_range[i] = max(tr1, tr2, tr3)
        
        # Calculate Directional Movement
        up_move = np.diff(high)
        down_move = -np.diff(low)
        
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        
        for i in range(1, len(high)):
            if up_move[i-1] > down_move[i-1] and up_move[i-1] > 0:
                plus_dm[i] = up_move[i-1]
            if down_move[i-1] > up_move[i-1] and down_move[i-1] > 0:
                minus_dm[i] = down_move[i-1]
        
        # Calculate ATR and ADX using simplified method
        atr_values = atr(high, low, close, period)
        
        # Calculate DI
        plus_di = np.zeros_like(close)
        minus_di = np.zeros_like(close)
        
        for i in range(period, len(close)):
            if atr_values[i] > 0:
                plus_di[i] = 100 * (np.mean(plus_dm[i-period+1:i+1]) / atr_values[i])
                minus_di[i] = 100 * (np.mean(minus_dm[i-period+1:i+1]) / atr_values[i])
        
        # Calculate ADX
        adx_values = np.zeros_like(close)
        for i in range(period, len(close)):
            if plus_di[i] + minus_di[i] > 0:
                dx = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
                adx_values[i] = np.mean([dx] * min(i-period+1, period))
        
        return adx_values


def williams_r(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], close: Union[pd.Series, np.ndarray], period: int = 14) -> Union[pd.Series, np.ndarray]:
    """
    Williams %R.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Period for calculation
        
    Returns:
        Williams %R values
    """
    if isinstance(high, pd.Series) and isinstance(low, pd.Series) and isinstance(close, pd.Series):
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        return -100 * ((highest_high - close) / (highest_high - lowest_low))
    else:
        williams = np.zeros_like(close)
        
        for i in range(period-1, len(close)):
            window_high = high[i-period+1:i+1]
            window_low = low[i-period+1:i+1]
            
            highest_high = np.max(window_high)
            lowest_low = np.min(window_low)
            
            if highest_high != lowest_low:
                williams[i] = -100 * ((highest_high - close[i]) / (highest_high - lowest_low))
            else:
                williams[i] = -50  # Default to -50 when no range
        
        return williams


def cci(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], close: Union[pd.Series, np.ndarray], period: int = 20) -> Union[pd.Series, np.ndarray]:
    """
    Commodity Channel Index (CCI).
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Period for calculation
        
    Returns:
        CCI values
    """
    if isinstance(high, pd.Series) and isinstance(low, pd.Series) and isinstance(close, pd.Series):
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        return (typical_price - sma_tp) / (0.015 * mad)
    else:
        typical_price = (high + low + close) / 3
        cci_values = np.zeros_like(close)
        
        for i in range(period-1, len(close)):
            window_tp = typical_price[i-period+1:i+1]
            sma_tp = np.mean(window_tp)
            mad = np.mean(np.abs(window_tp - sma_tp))
            
            if mad != 0:
                cci_values[i] = (typical_price[i] - sma_tp) / (0.015 * mad)
            else:
                cci_values[i] = 0  # Default to 0 when no deviation
        
        return cci_values


def roc(data: Union[pd.Series, np.ndarray], period: int = 12) -> Union[pd.Series, np.ndarray]:
    """
    Rate of Change (ROC).
    
    Args:
        data: Price data
        period: Period for calculation
        
    Returns:
        ROC values
    """
    if isinstance(data, pd.Series):
        return ((data - data.shift(period)) / data.shift(period)) * 100
    else:
        roc_values = np.zeros_like(data)
        for i in range(period, len(data)):
            if data[i-period] != 0:
                roc_values[i] = ((data[i] - data[i-period]) / data[i-period]) * 100
        return roc_values