"""
EMA RSI Strategy implementation for the quantitative trading framework.

This strategy combines Exponential Moving Average (EMA) crossovers with 
Relative Strength Index (RSI) for momentum confirmation.
"""

import pandas as pd
from typing import List, Dict, Any

from quant_framework.strategies.strategy_base import StrategyBase
from quant_framework.strategies.base.signal_types import Signal, SignalType, SignalStrength


class EmaRsiStrategy(StrategyBase):
    """
    EMA RSI Strategy.
    
    This strategy combines Exponential Moving Average (EMA) crossovers with 
    Relative Strength Index (RSI) for momentum confirmation.
    
    Core logic:
    1. Generate buy signals when fast EMA crosses above slow EMA and RSI is not overbought
    2. Generate sell signals when fast EMA crosses below slow EMA and RSI is not oversold
    """
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get the default configuration for the EMA RSI strategy.
        
        Returns:
            Default configuration dictionary with strategy-specific parameters
        """
        return {
            "parameters": {
                "ema_fast": 10,
                "ema_slow": 20,
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "rsi_neutral_high": 60,
                "rsi_neutral_low": 40,
                "signal_strength": SignalStrength.MODERATE.value
            },
            "max_data_rows": 10000,
            "symbols": []
        }
    
    def _initialize(self) -> None:
        """Initialize strategy-specific components."""
        # Set default parameters if not provided
        default_params = self.get_default_config().get("parameters", {})
        
        # Update with provided parameters
        for param, value in default_params.items():
            if param not in self.parameters:
                self.parameters[param] = value
        
        # Initialize state variables
        self.ema_fast_prev = {}
        self.ema_slow_prev = {}
        self.rsi_prev = {}
        self.indicators_cache = {}
    
    def calculate_signals(self, symbol: str, data: pd.DataFrame) -> List[Signal]:
        """
        Calculate trading signals for a symbol.
        
        Args:
            symbol: Trading symbol
            data: Historical market data
            
        Returns:
            List of trading signals
        """
        # Initialize signal list
        signals = []
        
        # Check if we have enough data
        min_bars = max(
            self.parameters.get("ema_slow", 20),
            self.parameters.get("rsi_period", 14)
        )
        
        if len(data) < min_bars:
            return signals
        
        # Initialize symbol-specific state if needed
        if symbol not in self.ema_fast_prev:
            self.ema_fast_prev[symbol] = None
            self.ema_slow_prev[symbol] = None
            self.rsi_prev[symbol] = None
            self.indicators_cache[symbol] = {}
        
        # Calculate indicators
        self._calculate_indicators(symbol, data)
        
        # Get current values
        current_price = data["close"].iloc[-1]
        indicators = self.indicators_cache[symbol]
        
        # Get previous values
        ema_fast_prev = self.ema_fast_prev[symbol]
        ema_slow_prev = self.ema_slow_prev[symbol]
        rsi_prev = self.rsi_prev[symbol]
        
        # Get current values
        ema_fast = indicators["ema_fast"]
        ema_slow = indicators["ema_slow"]
        rsi = indicators["rsi"]
        
        # Get parameters
        rsi_overbought = self.parameters.get("rsi_overbought", 70)
        rsi_oversold = self.parameters.get("rsi_oversold", 30)
        rsi_neutral_high = self.parameters.get("rsi_neutral_high", 60)
        rsi_neutral_low = self.parameters.get("rsi_neutral_low", 40)
        
        # Check for EMA crossover buy signal
        if (ema_fast_prev is not None and ema_slow_prev is not None and
            ema_fast_prev <= ema_slow_prev and ema_fast > ema_slow and
            rsi < rsi_overbought and rsi > rsi_neutral_low):
            
            signal_strength = SignalStrength(self.parameters.get("signal_strength", 2))
            
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=signal_strength,
                price=current_price,
                timestamp=data.index[-1],
                metadata={
                    "strategy": "ema_rsi",
                    "ema_fast": ema_fast,
                    "ema_slow": ema_slow,
                    "rsi": rsi,
                    "crossover": "bullish"
                }
            ))
        
        # Check for EMA crossover sell signal
        elif (ema_fast_prev is not None and ema_slow_prev is not None and
              ema_fast_prev >= ema_slow_prev and ema_fast < ema_slow and
              rsi > rsi_oversold and rsi < rsi_neutral_high):
            
            signal_strength = SignalStrength(self.parameters.get("signal_strength", 2))
            
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=signal_strength,
                price=current_price,
                timestamp=data.index[-1],
                metadata={
                    "strategy": "ema_rsi",
                    "ema_fast": ema_fast,
                    "ema_slow": ema_slow,
                    "rsi": rsi,
                    "crossover": "bearish"
                }
            ))
        
        # Update previous values
        self.ema_fast_prev[symbol] = ema_fast
        self.ema_slow_prev[symbol] = ema_slow
        self.rsi_prev[symbol] = rsi
        
        return signals
    
    def _calculate_indicators(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Calculate all technical indicators.
        
        Args:
            symbol: Trading symbol
            data: Price data
        """
        # Get parameters
        ema_fast_period = self.parameters.get("ema_fast", 10)
        ema_slow_period = self.parameters.get("ema_slow", 20)
        rsi_period = self.parameters.get("rsi_period", 14)
        
        # Calculate EMAs
        ema_fast = data["close"].ewm(span=ema_fast_period).mean().iloc[-1]
        ema_slow = data["close"].ewm(span=ema_slow_period).mean().iloc[-1]
        
        # Calculate RSI
        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.iloc[-1]
        
        # Store indicators in cache
        self.indicators_cache[symbol] = {
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "rsi": rsi
        }