"""
Mean Reversion Signal Generator Module

This module contains signal generation methods for mean reversion strategies.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

from quant_framework.strategies.signal_types import Signal, SignalType, SignalStrength
from quant_framework.strategies.mean_reversion_indicators import MeanReversionIndicators


class MeanReversionSignalGenerator:
    """
    Signal generator for mean reversion strategies.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize the signal generator.
        
        Args:
            parameters: Strategy parameters
        """
        self.parameters = parameters
        self.indicators_calculator = MeanReversionIndicators(parameters)
    
    def generate_signals(self, symbol: str, data: pd.DataFrame, current_position: float) -> List[Signal]:
        """
        Generate trading signals for a symbol.
        
        Args:
            symbol: Trading symbol
            data: Historical market data
            current_position: Current position size
            
        Returns:
            List of trading signals
        """
        # Calculate indicators
        self.indicators_calculator.calculate_indicators(symbol, data)
        indicators = self.indicators_calculator.get_indicators(symbol)
        
        signals = []
        
        # Generate signals based on enabled methods
        if self.parameters.get("use_bollinger_bands", True):
            signals.extend(self._generate_bollinger_band_signals(symbol, data, indicators, current_position))
        
        if self.parameters.get("use_rsi", False):
            signals.extend(self._generate_rsi_signals(symbol, data, indicators, current_position))
        
        if self.parameters.get("use_simple_mean_reversion", True):
            signals.extend(self._generate_simple_mean_reversion_signals(symbol, data, indicators, current_position))
        
        return signals
    
    def _generate_bollinger_band_signals(self, symbol: str, data: pd.DataFrame, 
                                         indicators: Dict[str, pd.Series], 
                                         current_position: float) -> List[Signal]:
        """
        Generate signals using Bollinger Bands.
        
        Args:
            symbol: Trading symbol
            data: Historical market data
            indicators: Technical indicators
            current_position: Current position size
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Check if we have enough data
        if 'bb_upper' not in indicators or 'bb_lower' not in indicators or 'bb_middle' not in indicators:
            return signals
        
        # Get latest values
        current_price = data['close'].iloc[-1]
        bb_upper = indicators['bb_upper'].iloc[-1]
        bb_lower = indicators['bb_lower'].iloc[-1]
        bb_middle = indicators['bb_middle'].iloc[-1]
        
        # Check if bands are widening (optional filter)
        if self.parameters.get("require_widening_bands", False):
            if not self.indicators_calculator.is_band_widening(symbol):
                return signals
        
        # Entry signals
        if current_position == 0:  # No position
            # Price touches or crosses below lower band - buy signal
            if current_price <= bb_lower:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.MODERATE,
                    price=current_price,
                    confidence=0.7,
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "mean_reversion",
                        "indicator": "bollinger_bands",
                        "bb_upper": bb_upper,
                        "bb_middle": bb_middle,
                        "bb_lower": bb_lower
                    }
                )
                signals.append(signal)
            
            # Price touches or crosses above upper band - sell signal
            elif current_price >= bb_upper:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.MODERATE,
                    price=current_price,
                    confidence=0.7,
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "mean_reversion",
                        "indicator": "bollinger_bands",
                        "bb_upper": bb_upper,
                        "bb_middle": bb_middle,
                        "bb_lower": bb_lower
                    }
                )
                signals.append(signal)
        
        # Exit signals
        elif current_position > 0:  # Long position
            # Price crosses back above middle band - exit signal
            if current_price >= bb_middle:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.WEAK,
                    price=current_price,
                    confidence=0.6,
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "mean_reversion",
                        "indicator": "bollinger_bands",
                        "action": "exit_long"
                    }
                )
                signals.append(signal)
        
        elif current_position < 0:  # Short position
            # Price crosses back below middle band - exit signal
            if current_price <= bb_middle:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.WEAK,
                    price=current_price,
                    confidence=0.6,
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "mean_reversion",
                        "indicator": "bollinger_bands",
                        "action": "exit_short"
                    }
                )
                signals.append(signal)
        
        return signals
    
    def _generate_rsi_signals(self, symbol: str, data: pd.DataFrame, 
                             indicators: Dict[str, pd.Series], 
                             current_position: float) -> List[Signal]:
        """
        Generate signals using RSI.
        
        Args:
            symbol: Trading symbol
            data: Historical market data
            indicators: Technical indicators
            current_position: Current position size
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Check if we have RSI data
        if 'rsi' not in indicators:
            return signals
        
        # Get latest values
        current_price = data['close'].iloc[-1]
        rsi = indicators['rsi'].iloc[-1]
        oversold_threshold = self.parameters.get("rsi_oversold", 30)
        overbought_threshold = self.parameters.get("rsi_overbought", 70)
        
        # Entry signals
        if current_position == 0:  # No position
            # RSI crosses below oversold threshold - buy signal
            if rsi < oversold_threshold:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.MODERATE,
                    price=current_price,
                    confidence=min(1.0, (oversold_threshold - rsi) / 20),  # Higher confidence for lower RSI
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "mean_reversion",
                        "indicator": "rsi",
                        "rsi": rsi
                    }
                )
                signals.append(signal)
            
            # RSI crosses above overbought threshold - sell signal
            elif rsi > overbought_threshold:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.MODERATE,
                    price=current_price,
                    confidence=min(1.0, (rsi - overbought_threshold) / 20),  # Higher confidence for higher RSI
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "mean_reversion",
                        "indicator": "rsi",
                        "rsi": rsi
                    }
                )
                signals.append(signal)
        
        # Exit signals
        elif current_position > 0:  # Long position
            # RSI crosses back above 50 - exit signal
            if rsi > 50:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.WEAK,
                    price=current_price,
                    confidence=0.6,
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "mean_reversion",
                        "indicator": "rsi",
                        "action": "exit_long"
                    }
                )
                signals.append(signal)
        
        elif current_position < 0:  # Short position
            # RSI crosses back below 50 - exit signal
            if rsi < 50:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.WEAK,
                    price=current_price,
                    confidence=0.6,
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "mean_reversion",
                        "indicator": "rsi",
                        "action": "exit_short"
                    }
                )
                signals.append(signal)
        
        return signals
    
    def _generate_simple_mean_reversion_signals(self, symbol: str, data: pd.DataFrame, 
                                                indicators: Dict[str, pd.Series], 
                                                current_position: float) -> List[Signal]:
        """
        Generate signals using simple mean reversion (z-score).
        
        Args:
            symbol: Trading symbol
            data: Historical market data
            indicators: Technical indicators
            current_position: Current position size
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Check if we have z-score data
        if 'zscore' not in indicators:
            return signals
        
        # Get latest values
        current_price = data['close'].iloc[-1]
        zscore = indicators['zscore'].iloc[-1]
        entry_threshold = self.parameters.get("entry_threshold", 2.0)
        exit_threshold = self.parameters.get("exit_threshold", 0.5)
        
        # Entry signals
        if current_position == 0:  # No position
            # Price significantly below mean - buy signal
            if zscore < -entry_threshold:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.MODERATE,
                    price=current_price,
                    confidence=min(1.0, abs(zscore) / entry_threshold),  # Higher confidence for larger deviation
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "mean_reversion",
                        "indicator": "zscore",
                        "zscore": zscore
                    }
                )
                signals.append(signal)
            
            # Price significantly above mean - sell signal
            elif zscore > entry_threshold:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.MODERATE,
                    price=current_price,
                    confidence=min(1.0, abs(zscore) / entry_threshold),  # Higher confidence for larger deviation
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "mean_reversion",
                        "indicator": "zscore",
                        "zscore": zscore
                    }
                )
                signals.append(signal)
        
        # Exit signals
        elif current_position > 0:  # Long position
            # Price reverts to mean - exit signal
            if zscore > -exit_threshold:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.WEAK,
                    price=current_price,
                    confidence=0.7,
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "mean_reversion",
                        "indicator": "zscore",
                        "action": "exit_long",
                        "zscore": zscore
                    }
                )
                signals.append(signal)
        
        elif current_position < 0:  # Short position
            # Price reverts to mean - exit signal
            if zscore < exit_threshold:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.WEAK,
                    price=current_price,
                    confidence=0.7,
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "mean_reversion",
                        "indicator": "zscore",
                        "action": "exit_short",
                        "zscore": zscore
                    }
                )
                signals.append(signal)
        
        return signals