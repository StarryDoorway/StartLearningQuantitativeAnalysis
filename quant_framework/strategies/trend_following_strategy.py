"""
Trend Following Strategy implementation for the quantitative trading framework.

This strategy identifies and follows market trends using multiple technical indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

from .strategy_base import StrategyBase, Signal, SignalType, SignalStrength
from ..utils.config_loader import get_config
from ..core.event_bus import get_event_bus, EventType, Event


class TrendFollowingStrategy(StrategyBase):
    """
    Trend Following Strategy.
    
    This strategy identifies and follows market trends using multiple technical indicators
    including EMA crossovers, Bollinger Bands, MACD, and channel breakouts.
    
    Core logic:
    1. Buy when multiple indicators confirm an uptrend
    2. Sell when multiple indicators confirm a downtrend
    """
    
    def _initialize(self) -> None:
        """Initialize strategy-specific components."""
        # Subscribe to market data events
        self.event_bus.subscribe(event_type=EventType.MARKET_DATA, callback=self.on_bar)
        
        # Subscribe to portfolio events
        self.event_bus.subscribe(event_type=EventType.PORTFOLIO_UPDATE, callback=self._on_portfolio_update)
        
        # Set default parameters if not provided
        default_params = {
            "ema_fast": 20,
            "ema_slow": 50,
            "bollinger_period": 20,
            "bollinger_dev": 2.0,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "channel_length": 20,
            "signal_combination": "any",  # 'any', 'all', 'ema_bollinger', 'ema_macd', 'bollinger_macd'
            "confirm_threshold": 1,  # Number of bars to confirm signal
            "signal_strength": SignalStrength.MODERATE.value
        }
        
        # Update with provided parameters
        for param, value in default_params.items():
            if param not in self.parameters:
                self.parameters[param] = value
        
        # Initialize state variables
        self.buy_signal_count = {}
        self.sell_signal_count = {}
        self.indicators_cache = {}
    
    def _on_portfolio_update(self, event) -> None:
        """
        Handle portfolio update events.
        
        Args:
            event: Portfolio update event
        """
        # Update current positions
        if "positions" in event.data:
            self.current_positions = event.data["positions"]
    
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
            self.parameters.get("ema_slow", 50),
            self.parameters.get("bollinger_period", 20),
            self.parameters.get("macd_slow", 26),
            self.parameters.get("channel_length", 20)
        )
        
        if len(data) < min_bars:
            return signals
        
        # Initialize symbol-specific state if needed
        if symbol not in self.buy_signal_count:
            self.buy_signal_count[symbol] = 0
            self.sell_signal_count[symbol] = 0
            self.indicators_cache[symbol] = {}
        
        # Calculate indicators
        self._calculate_indicators(symbol, data)
        
        # Get current values
        current_price = data["close"].iloc[-1]
        indicators = self.indicators_cache[symbol]
        
        # Calculate buy signal
        buy_signal = self._calculate_buy_signal(symbol, indicators)
        if buy_signal:
            self.buy_signal_count[symbol] += 1
        else:
            self.buy_signal_count[symbol] = 0
        
        # Calculate sell signal
        sell_signal = self._calculate_sell_signal(symbol, indicators)
        if sell_signal:
            self.sell_signal_count[symbol] += 1
        else:
            self.sell_signal_count[symbol] = 0
        
        # Generate buy signal if confirmed
        confirm_threshold = self.parameters.get("confirm_threshold", 1)
        if self.buy_signal_count[symbol] >= confirm_threshold:
            signal_strength = SignalStrength(self.parameters.get("signal_strength", 2))
            
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=signal_strength,
                price=current_price,
                timestamp=data.index[-1],
                metadata={
                    "strategy": "trend_following",
                    "indicators": {
                        "ema_crossover": indicators["ema_crossover"],
                        "bollinger_position": indicators["bollinger_position"],
                        "macd_position": indicators["macd_position"],
                        "channel_position": indicators["channel_position"]
                    }
                }
            ))
            
            # Reset counter
            self.buy_signal_count[symbol] = 0
        
        # Generate sell signal if confirmed
        if self.sell_signal_count[symbol] >= confirm_threshold:
            signal_strength = SignalStrength(self.parameters.get("signal_strength", 2))
            
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=signal_strength,
                price=current_price,
                timestamp=data.index[-1],
                metadata={
                    "strategy": "trend_following",
                    "indicators": {
                        "ema_crossover": indicators["ema_crossover"],
                        "bollinger_position": indicators["bollinger_position"],
                        "macd_position": indicators["macd_position"],
                        "channel_position": indicators["channel_position"]
                    }
                }
            ))
            
            # Reset counter
            self.sell_signal_count[symbol] = 0
        
        return signals
    
    def _calculate_indicators(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Calculate all technical indicators.
        
        Args:
            symbol: Trading symbol
            data: Price data
        """
        # Get parameters
        ema_fast = self.parameters.get("ema_fast", 20)
        ema_slow = self.parameters.get("ema_slow", 50)
        bollinger_period = self.parameters.get("bollinger_period", 20)
        bollinger_dev = self.parameters.get("bollinger_dev", 2.0)
        macd_fast = self.parameters.get("macd_fast", 12)
        macd_slow = self.parameters.get("macd_slow", 26)
        macd_signal = self.parameters.get("macd_signal", 9)
        channel_length = self.parameters.get("channel_length", 20)
        
        # Calculate EMA
        ema_fast_values = data["close"].ewm(span=ema_fast).mean()
        ema_slow_values = data["close"].ewm(span=ema_slow).mean()
        
        # Calculate EMA crossover (1 if fast > slow, -1 if fast < slow, 0 otherwise)
        ema_crossover = np.where(ema_fast_values.iloc[-1] > ema_slow_values.iloc[-1], 1, -1)
        
        # Calculate Bollinger Bands
        sma = data["close"].rolling(window=bollinger_period).mean()
        std = data["close"].rolling(window=bollinger_period).std()
        upper_band = sma + (std * bollinger_dev)
        lower_band = sma - (std * bollinger_dev)
        
        # Calculate Bollinger Band position (1 if above upper, -1 if below lower, 0 otherwise)
        current_price = data["close"].iloc[-1]
        bollinger_position = 0
        if current_price > upper_band.iloc[-1]:
            bollinger_position = 1
        elif current_price < lower_band.iloc[-1]:
            bollinger_position = -1
        
        # Calculate MACD
        ema_macd_fast = data["close"].ewm(span=macd_fast).mean()
        ema_macd_slow = data["close"].ewm(span=macd_slow).mean()
        macd_line = ema_macd_fast - ema_macd_slow
        signal_line = macd_line.ewm(span=macd_signal).mean()
        
        # Calculate MACD position (1 if MACD > signal, -1 if MACD < signal, 0 otherwise)
        macd_position = 0
        if macd_line.iloc[-1] > signal_line.iloc[-1]:
            macd_position = 1
        elif macd_line.iloc[-1] < signal_line.iloc[-1]:
            macd_position = -1
        
        # Calculate Channel position
        highest = data["high"].rolling(window=channel_length).max()
        lowest = data["low"].rolling(window=channel_length).min()
        
        # Calculate channel position (1 if above high, -1 if below low, 0 otherwise)
        channel_position = 0
        if current_price > highest.iloc[-2]:  # Use previous high to avoid lookahead bias
            channel_position = 1
        elif current_price < lowest.iloc[-2]:  # Use previous low to avoid lookahead bias
            channel_position = -1
        
        # Store indicators in cache
        self.indicators_cache[symbol] = {
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
    
    def _calculate_buy_signal(self, symbol: str, indicators: Dict[str, Any]) -> bool:
        """
        Calculate buy signal based on indicators.
        
        Args:
            symbol: Trading symbol
            indicators: Technical indicators
            
        Returns:
            True if buy signal is detected
        """
        # Get signal combination method
        signal_combination = self.parameters.get("signal_combination", "any")
        
        # Extract individual signals
        ema_buy = indicators["ema_crossover"] > 0
        bollinger_buy = indicators["bollinger_position"] > 0
        macd_buy = indicators["macd_position"] > 0
        channel_buy = indicators["channel_position"] > 0
        
        # Combine signals based on method
        if signal_combination == "any":
            return ema_buy or bollinger_buy or macd_buy or channel_buy
        elif signal_combination == "all":
            return ema_buy and bollinger_buy and macd_buy and channel_buy
        elif signal_combination == "ema_bollinger":
            return ema_buy and bollinger_buy
        elif signal_combination == "ema_macd":
            return ema_buy and macd_buy
        elif signal_combination == "bollinger_macd":
            return bollinger_buy and macd_buy
        else:
            return ema_buy
    
    def _calculate_sell_signal(self, symbol: str, indicators: Dict[str, Any]) -> bool:
        """
        Calculate sell signal based on indicators.
        
        Args:
            symbol: Trading symbol
            indicators: Technical indicators
            
        Returns:
            True if sell signal is detected
        """
        # Get signal combination method
        signal_combination = self.parameters.get("signal_combination", "any")
        
        # Extract individual signals
        ema_sell = indicators["ema_crossover"] < 0
        bollinger_sell = indicators["bollinger_position"] < 0
        macd_sell = indicators["macd_position"] < 0
        channel_sell = indicators["channel_position"] < 0
        
        # Combine signals based on method
        if signal_combination == "any":
            return ema_sell or bollinger_sell or macd_sell or channel_sell
        elif signal_combination == "all":
            return ema_sell and bollinger_sell and macd_sell and channel_sell
        elif signal_combination == "ema_bollinger":
            return ema_sell and bollinger_sell
        elif signal_combination == "ema_macd":
            return ema_sell and macd_sell
        elif signal_combination == "bollinger_macd":
            return bollinger_sell and macd_sell
        else:
            return ema_sell