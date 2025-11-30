"""
Trend Following Signal Generator module for the quantitative trading framework.

This module provides signal generation logic for trend following strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from quant_framework.strategies.strategy_base import Signal, SignalType
from quant_framework.strategies.indicators.trend_following_indicators import TrendFollowingIndicators


class TrendFollowingSignalGenerator:
    """
    Trend Following Signal Generator class.
    
    This class provides methods to generate trading signals based on trend following indicators.
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the signal generator.
        
        Args:
            params: Signal generation parameters
        """
        self.params = params
        self.indicators = TrendFollowingIndicators()
        
        # Default parameters
        self.confirm_bars = params.get("confirm_bars", 3)
        self.signal_method = params.get("signal_method", "any")
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        """
        Generate trading signals based on trend following indicators.
        
        Args:
            data: OHLCV data
            symbol: Trading symbol
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Calculate all indicators
        indicators = self.indicators.calculate_all_indicators(data, self.params)
        
        # Generate buy signal
        buy_signal = self._calculate_buy_signal(indicators, symbol, data)
        if buy_signal:
            signals.append(buy_signal)
        
        # Generate sell signal
        sell_signal = self._calculate_sell_signal(indicators, symbol, data)
        if sell_signal:
            signals.append(sell_signal)
        
        return signals
    
    def _calculate_buy_signal(self, indicators: Dict[str, Any], symbol: str, data: pd.DataFrame) -> Optional[Signal]:
        """
        Calculate buy signal based on indicators.
        
        Args:
            indicators: Dictionary of indicator values
            symbol: Trading symbol
            data: OHLCV data
            
        Returns:
            Buy signal or None
        """
        # Get current price
        current_price = data["close"].iloc[-1]
        current_time = data.index[-1]
        
        # Initialize signal strength
        signal_strength = 0
        
        # Check EMA crossover
        if indicators["ema_crossover"] > 0:
            signal_strength += 1
        
        # Check Bollinger Band position
        if indicators["bollinger_position"] > 0:
            signal_strength += 1
        
        # Check MACD position
        if indicators["macd_position"] > 0:
            signal_strength += 1
        
        # Check channel position
        if indicators["channel_position"] > 0:
            signal_strength += 1
        
        # Generate signal based on signal method
        if self.signal_method == "any" and signal_strength > 0:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=signal_strength / 4.0,
                price=current_price,
                timestamp=current_time,
                quantity=1.0,
                metadata={
                    "ema_crossover": indicators["ema_crossover"],
                    "bollinger_position": indicators["bollinger_position"],
                    "macd_position": indicators["macd_position"],
                    "channel_position": indicators["channel_position"],
                    "signal_method": self.signal_method
                }
            )
        elif self.signal_method == "all" and signal_strength == 4:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=1.0,
                price=current_price,
                timestamp=current_time,
                quantity=1.0,
                metadata={
                    "ema_crossover": indicators["ema_crossover"],
                    "bollinger_position": indicators["bollinger_position"],
                    "macd_position": indicators["macd_position"],
                    "channel_position": indicators["channel_position"],
                    "signal_method": self.signal_method
                }
            )
        elif self.signal_method == "ema_bollinger" and indicators["ema_crossover"] > 0 and indicators["bollinger_position"] > 0:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=0.8,
                price=current_price,
                timestamp=current_time,
                quantity=1.0,
                metadata={
                    "ema_crossover": indicators["ema_crossover"],
                    "bollinger_position": indicators["bollinger_position"],
                    "signal_method": self.signal_method
                }
            )
        
        return None
    
    def _calculate_sell_signal(self, indicators: Dict[str, Any], symbol: str, data: pd.DataFrame) -> Optional[Signal]:
        """
        Calculate sell signal based on indicators.
        
        Args:
            indicators: Dictionary of indicator values
            symbol: Trading symbol
            data: OHLCV data
            
        Returns:
            Sell signal or None
        """
        # Get current price
        current_price = data["close"].iloc[-1]
        current_time = data.index[-1]
        
        # Initialize signal strength
        signal_strength = 0
        
        # Check EMA crossover
        if indicators["ema_crossover"] < 0:
            signal_strength += 1
        
        # Check Bollinger Band position
        if indicators["bollinger_position"] < 0:
            signal_strength += 1
        
        # Check MACD position
        if indicators["macd_position"] < 0:
            signal_strength += 1
        
        # Check channel position
        if indicators["channel_position"] < 0:
            signal_strength += 1
        
        # Generate signal based on signal method
        if self.signal_method == "any" and signal_strength > 0:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=signal_strength / 4.0,
                price=current_price,
                timestamp=current_time,
                quantity=1.0,
                metadata={
                    "ema_crossover": indicators["ema_crossover"],
                    "bollinger_position": indicators["bollinger_position"],
                    "macd_position": indicators["macd_position"],
                    "channel_position": indicators["channel_position"],
                    "signal_method": self.signal_method
                }
            )
        elif self.signal_method == "all" and signal_strength == 4:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=1.0,
                price=current_price,
                timestamp=current_time,
                quantity=1.0,
                metadata={
                    "ema_crossover": indicators["ema_crossover"],
                    "bollinger_position": indicators["bollinger_position"],
                    "macd_position": indicators["macd_position"],
                    "channel_position": indicators["channel_position"],
                    "signal_method": self.signal_method
                }
            )
        elif self.signal_method == "ema_bollinger" and indicators["ema_crossover"] < 0 and indicators["bollinger_position"] < 0:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=0.8,
                price=current_price,
                timestamp=current_time,
                quantity=1.0,
                metadata={
                    "ema_crossover": indicators["ema_crossover"],
                    "bollinger_position": indicators["bollinger_position"],
                    "signal_method": self.signal_method
                }
            )
        
        return None