"""
RSI Divergence Signal Generator module for the quantitative trading framework.

This module contains the signal generation logic for RSI divergence trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from .strategy_base import Signal, SignalType, SignalStrength
from .rsi_divergence_indicators import RsiDivergenceIndicators


class RsiDivergenceSignalGenerator:
    """
    Class for generating trading signals based on RSI divergence patterns.
    
    This class provides methods to detect bullish and bearish divergences
    between price action and the RSI indicator.
    """
    
    def __init__(self, indicators_calculator: RsiDivergenceIndicators):
        """
        Initialize the RSI Divergence Signal Generator.
        
        Args:
            indicators_calculator: Instance of RsiDivergenceIndicators
        """
        self.indicators = indicators_calculator
        
        # Initialize signal counters
        self.buy_signal_count = {}
        self.sell_signal_count = {}
    
    def generate_signals(self, symbol: str, data: pd.DataFrame, 
                        params: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals based on RSI divergence.
        
        Args:
            symbol: Trading symbol
            data: Historical market data
            params: Strategy parameters
            
        Returns:
            List of trading signals
        """
        # Initialize signal list
        signals = []
        
        # Check if we have enough data
        min_bars = params.get("min_trade_bars", 20)
        if len(data) < min_bars:
            return signals
        
        # Initialize symbol-specific state if needed
        if symbol not in self.buy_signal_count:
            self.buy_signal_count[symbol] = 0
            self.sell_signal_count[symbol] = 0
        
        # Calculate indicators
        indicators = self.indicators.calculate_indicators(symbol, data, params)
        rsi = indicators["rsi"]
        current_price = indicators["current_price"]
        current_rsi = indicators["current_rsi"]
        
        # Get parameters
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        divergence_lookback = params.get("divergence_lookback", 5)
        price_change_threshold = params.get("price_change_threshold", 0.005)
        rsi_change_threshold = params.get("rsi_change_threshold", 2.0)
        
        # Check for bullish divergence (buy signal)
        buy_signal = self.indicators.check_bullish_divergence(
            symbol, data, rsi, rsi_oversold, divergence_lookback,
            price_change_threshold, rsi_change_threshold
        )
        
        if buy_signal:
            self.buy_signal_count[symbol] += 1
        else:
            self.buy_signal_count[symbol] = 0
        
        # Check for bearish divergence (sell signal)
        sell_signal = self.indicators.check_bearish_divergence(
            symbol, data, rsi, rsi_overbought, divergence_lookback,
            price_change_threshold, rsi_change_threshold
        )
        
        if sell_signal:
            self.sell_signal_count[symbol] += 1
        else:
            self.sell_signal_count[symbol] = 0
        
        # Generate buy signal if confirmed
        confirm_bars = params.get("confirm_bars", 1)
        if self.buy_signal_count[symbol] >= confirm_bars:
            signal_strength = SignalStrength(params.get("signal_strength", 2))
            
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=signal_strength,
                price=current_price,
                timestamp=data.index[-1],
                metadata={
                    "strategy": "rsi_divergence",
                    "rsi": current_rsi,
                    "divergence_type": "bullish"
                }
            ))
            
            # Reset counter
            self.buy_signal_count[symbol] = 0
        
        # Generate sell signal if confirmed
        if self.sell_signal_count[symbol] >= confirm_bars:
            signal_strength = SignalStrength(params.get("signal_strength", 2))
            
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=signal_strength,
                price=current_price,
                timestamp=data.index[-1],
                metadata={
                    "strategy": "rsi_divergence",
                    "rsi": current_rsi,
                    "divergence_type": "bearish"
                }
            ))
            
            # Reset counter
            self.sell_signal_count[symbol] = 0
        
        return signals
    
    def reset_signal_counters(self, symbol: str) -> None:
        """
        Reset signal counters for a symbol.
        
        Args:
            symbol: Trading symbol
        """
        if symbol in self.buy_signal_count:
            self.buy_signal_count[symbol] = 0
        if symbol in self.sell_signal_count:
            self.sell_signal_count[symbol] = 0
    
    def get_signal_counts(self, symbol: str) -> Dict[str, int]:
        """
        Get current signal counts for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with buy and sell signal counts
        """
        return {
            "buy_signal_count": self.buy_signal_count.get(symbol, 0),
            "sell_signal_count": self.sell_signal_count.get(symbol, 0)
        }