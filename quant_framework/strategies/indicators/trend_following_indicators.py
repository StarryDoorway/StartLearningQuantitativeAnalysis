"""
Trend Following Indicators module for the quantitative trading framework.

This module provides technical indicators for trend following strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


class TrendFollowingIndicators:
    """
    Trend Following Indicators class.
    
    This class provides methods to calculate various technical indicators
    used in trend following strategies.
    """
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).
        
        Args:
            data: Price data
            period: EMA period
            
        Returns:
            EMA values
        """
        return data.ewm(span=period).mean()
    
    @staticmethod
    def calculate_ema_crossover(data: pd.Series, fast_period: int, slow_period: int) -> int:
        """
        Calculate EMA crossover signal.
        
        Args:
            data: Price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            
        Returns:
            1 if fast EMA > slow EMA, -1 if fast EMA < slow EMA
        """
        ema_fast = TrendFollowingIndicators.calculate_ema(data, fast_period)
        ema_slow = TrendFollowingIndicators.calculate_ema(data, slow_period)
        
        if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
            return 1
        else:
            return -1
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int, dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: Price data
            period: Period for moving average and standard deviation
            dev: Number of standard deviations
            
        Returns:
            Tuple of (middle band, upper band, lower band)
        """
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std * dev)
        lower_band = sma - (std * dev)
        
        return sma, upper_band, lower_band
    
    @staticmethod
    def calculate_bollinger_position(price: float, upper_band: float, lower_band: float) -> int:
        """
        Calculate position relative to Bollinger Bands.
        
        Args:
            price: Current price
            upper_band: Upper band value
            lower_band: Lower band value
            
        Returns:
            1 if price > upper band, -1 if price < lower band, 0 otherwise
        """
        if price > upper_band:
            return 1
        elif price < lower_band:
            return -1
        else:
            return 0
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast_period: int, slow_period: int, signal_period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            Tuple of (MACD line, signal line, histogram)
        """
        ema_fast = TrendFollowingIndicators.calculate_ema(data, fast_period)
        ema_slow = TrendFollowingIndicators.calculate_ema(data, slow_period)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_macd_position(macd_line: float, signal_line: float) -> int:
        """
        Calculate MACD position relative to signal line.
        
        Args:
            macd_line: MACD line value
            signal_line: Signal line value
            
        Returns:
            1 if MACD > signal, -1 if MACD < signal, 0 otherwise
        """
        if macd_line > signal_line:
            return 1
        elif macd_line < signal_line:
            return -1
        else:
            return 0
    
    @staticmethod
    def calculate_channel(high: pd.Series, low: pd.Series, period: int) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate price channel (highest high and lowest low).
        
        Args:
            high: High price data
            low: Low price data
            period: Channel period
            
        Returns:
            Tuple of (highest high, lowest low)
        """
        highest = high.rolling(window=period).max()
        lowest = low.rolling(window=period).min()
        
        return highest, lowest
    
    @staticmethod
    def calculate_channel_position(price: float, highest: float, lowest: float) -> int:
        """
        Calculate position relative to price channel.
        
        Args:
            price: Current price
            highest: Highest high value
            lowest: Lowest low value
            
        Returns:
            1 if price > highest, -1 if price < lowest, 0 otherwise
        """
        if price > highest:
            return 1
        elif price < lowest:
            return -1
        else:
            return 0
    
    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate all trend following indicators.
        
        Args:
            data: OHLCV data
            params: Indicator parameters
            
        Returns:
            Dictionary of all indicators
        """
        # Get parameters
        ema_fast = params.get("ema_fast", 20)
        ema_slow = params.get("ema_slow", 50)
        bollinger_period = params.get("bollinger_period", 20)
        bollinger_dev = params.get("bollinger_dev", 2.0)
        macd_fast = params.get("macd_fast", 12)
        macd_slow = params.get("macd_slow", 26)
        macd_signal = params.get("macd_signal", 9)
        channel_length = params.get("channel_length", 20)
        
        # Calculate EMA
        ema_fast_values = TrendFollowingIndicators.calculate_ema(data["close"], ema_fast)
        ema_slow_values = TrendFollowingIndicators.calculate_ema(data["close"], ema_slow)
        
        # Calculate EMA crossover
        ema_crossover = TrendFollowingIndicators.calculate_ema_crossover(data["close"], ema_fast, ema_slow)
        
        # Calculate Bollinger Bands
        _, upper_band, lower_band = TrendFollowingIndicators.calculate_bollinger_bands(
            data["close"], bollinger_period, bollinger_dev
        )
        
        # Calculate Bollinger Band position
        current_price = data["close"].iloc[-1]
        bollinger_position = TrendFollowingIndicators.calculate_bollinger_position(
            current_price, upper_band.iloc[-1], lower_band.iloc[-1]
        )
        
        # Calculate MACD
        macd_line, signal_line, _ = TrendFollowingIndicators.calculate_macd(
            data["close"], macd_fast, macd_slow, macd_signal
        )
        
        # Calculate MACD position
        macd_position = TrendFollowingIndicators.calculate_macd_position(
            macd_line.iloc[-1], signal_line.iloc[-1]
        )
        
        # Calculate Channel
        highest, lowest = TrendFollowingIndicators.calculate_channel(
            data["high"], data["low"], channel_length
        )
        
        # Calculate channel position (use previous values to avoid lookahead bias)
        channel_position = TrendFollowingIndicators.calculate_channel_position(
            current_price, highest.iloc[-2], lowest.iloc[-2]
        )
        
        # Return all indicators
        return {
            "ema_fast": ema_fast_values.iloc[-1],
            "ema_slow": ema_slow_values.iloc[-1],
            "ema_crossover": ema_crossover,
            "bollinger_upper": upper_band.iloc[-1],
            "bollinger_lower": lower_band.iloc[-1],
            "bollinger_position": bollinger_position,
            "macd_line": macd_line.iloc[-1],
            "macd_signal": signal_line.iloc[-1],
            "macd_position": macd_position,
            "channel_high": highest.iloc[-2],
            "channel_low": lowest.iloc[-2],
            "channel_position": channel_position
        }