"""
RSI Divergence Strategy implementation for the quantitative trading framework.

This strategy identifies potential trend reversals by detecting divergences
between price action and the RSI indicator.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

from .strategy_base import StrategyBase, Signal, SignalType, SignalStrength
from ..utils.config_loader import get_config
from ..core.event_bus import get_event_bus, EventType, Event


class RsiDivergenceStrategy(StrategyBase):
    """
    RSI Divergence Strategy.
    
    This strategy identifies potential trend reversals by detecting divergences
    between price action and the RSI indicator.
    
    Core logic:
    1. Bullish divergence: When RSI makes lower lows while price makes higher lows
    2. Bearish divergence: When RSI makes higher highs while price makes lower highs
    """
    
    def _initialize(self) -> None:
        """Initialize strategy-specific components."""
        # Subscribe to market data events
        self.event_bus.subscribe(event_type=EventType.MARKET_DATA, callback=self.on_bar)
        
        # Subscribe to portfolio events
        self.event_bus.subscribe(event_type=EventType.PORTFOLIO_UPDATE, callback=self._on_portfolio_update)
        
        # Set default parameters if not provided
        default_params = {
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "divergence_lookback": 5,
            "price_change_threshold": 0.005,
            "rsi_change_threshold": 2.0,
            "confirm_bars": 1,
            "min_trade_bars": 20,
            "signal_strength": SignalStrength.MODERATE.value
        }
        
        # Update with provided parameters
        for param, value in default_params.items():
            if param not in self.parameters:
                self.parameters[param] = value
        
        # Initialize state variables
        self.last_rsi_low = {}
        self.last_price_low = {}
        self.last_rsi_high = {}
        self.last_price_high = {}
        self.buy_signal_count = {}
        self.sell_signal_count = {}
    
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
        min_bars = self.parameters.get("min_trade_bars", 20)
        if len(data) < min_bars:
            return signals
        
        # Initialize symbol-specific state if needed
        if symbol not in self.last_rsi_low:
            self.last_rsi_low[symbol] = None
            self.last_price_low[symbol] = None
            self.last_rsi_high[symbol] = None
            self.last_price_high[symbol] = None
            self.buy_signal_count[symbol] = 0
            self.sell_signal_count[symbol] = 0
        
        # Calculate indicators
        rsi_period = self.parameters.get("rsi_period", 14)
        rsi = self._calculate_rsi(data["close"], rsi_period)
        
        # Update extremes for divergence detection
        self._update_extremes(symbol, data, rsi)
        
        # Get current values
        current_price = data["close"].iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        # Add debug logging
        rsi_overbought = self.parameters.get("rsi_overbought", 70)
        rsi_oversold = self.parameters.get("rsi_oversold", 30)
        self.logger.debug(f"{symbol}: Current RSI = {current_rsi:.2f}, Overbought={rsi_overbought}, Oversold={rsi_oversold}")
        
        # Check for bullish divergence (buy signal)
        buy_signal = self._check_bullish_divergence(symbol, data, rsi)
        if buy_signal:
            self.buy_signal_count[symbol] += 1
            self.logger.debug(f"{symbol}: Bullish divergence detected, count={self.buy_signal_count[symbol]}")
        else:
            self.buy_signal_count[symbol] = 0
        
        # Check for bearish divergence (sell signal)
        sell_signal = self._check_bearish_divergence(symbol, data, rsi)
        if sell_signal:
            self.sell_signal_count[symbol] += 1
            self.logger.debug(f"{symbol}: Bearish divergence detected, count={self.sell_signal_count[symbol]}")
        else:
            self.sell_signal_count[symbol] = 0
        
        # Generate buy signal if confirmed
        confirm_bars = self.parameters.get("confirm_bars", 1)
        if self.buy_signal_count[symbol] >= confirm_bars:
            signal_strength = SignalStrength(self.parameters.get("signal_strength", 2))
            
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
            signal_strength = SignalStrength(self.parameters.get("signal_strength", 2))
            
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
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI indicator.
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI series
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _update_extremes(self, symbol: str, data: pd.DataFrame, rsi: pd.Series) -> None:
        """
        Update RSI and price extremes for divergence detection.
        
        Args:
            symbol: Trading symbol
            data: Price data
            rsi: RSI values
        """
        current_rsi = rsi.iloc[-1]
        current_price = data["close"].iloc[-1]
        
        # Update RSI and price lows
        if self.last_rsi_low[symbol] is None or current_rsi < self.last_rsi_low[symbol]:
            self.last_rsi_low[symbol] = current_rsi
            self.last_price_low[symbol] = current_price
        
        # Update RSI and price highs
        if self.last_rsi_high[symbol] is None or current_rsi > self.last_rsi_high[symbol]:
            self.last_rsi_high[symbol] = current_rsi
            self.last_price_high[symbol] = current_price
    
    def _check_bullish_divergence(self, symbol: str, data: pd.DataFrame, rsi: pd.Series) -> bool:
        """
        Check for bullish divergence (buy signal).
        
        Args:
            symbol: Trading symbol
            data: Price data
            rsi: RSI values
            
        Returns:
            True if bullish divergence is detected
        """
        rsi_oversold = self.parameters.get("rsi_oversold", 30)
        divergence_lookback = self.parameters.get("divergence_lookback", 5)
        price_change_threshold = self.parameters.get("price_change_threshold", 0.005)
        rsi_change_threshold = self.parameters.get("rsi_change_threshold", 2.0)
        
        # RSI must be in oversold territory
        if rsi.iloc[-1] >= rsi_oversold:
            self.logger.debug(f"{symbol}: RSI {rsi.iloc[-1]:.2f} not in oversold territory (>= {rsi_oversold})")
            return False
        
        # Check if we have enough data
        if len(data) < divergence_lookback + 1 or len(rsi) < divergence_lookback + 1:
            self.logger.debug(f"{symbol}: Not enough data for divergence check")
            return False
        
        # Get past values
        past_prices = data["close"].iloc[-(divergence_lookback+1):-1]
        past_rsi = rsi.iloc[-(divergence_lookback+1):-1]
        
        # Check if RSI is decreasing
        rsi_decreasing = all(past_rsi.iloc[i] > past_rsi.iloc[i+1] for i in range(len(past_rsi)-1))
        rsi_still_decreasing = past_rsi.iloc[-1] > rsi.iloc[-1]
        
        # Check if price is not decreasing significantly
        price_change = (data["close"].iloc[-1] - past_prices.iloc[-1]) / past_prices.iloc[-1]
        price_not_decreasing = price_change > -price_change_threshold
        
        # Check if RSI change is significant
        rsi_change = past_rsi.iloc[0] - rsi.iloc[-1]
        rsi_significant_change = rsi_change > rsi_change_threshold
        
        # Log debug info
        self.logger.debug(f"{symbol}: Bullish check - RSI decreasing: {rsi_decreasing}, still decreasing: {rsi_still_decreasing}")
        self.logger.debug(f"{symbol}: Bullish check - Price change: {price_change:.4f}, not decreasing: {price_not_decreasing}")
        self.logger.debug(f"{symbol}: Bullish check - RSI change: {rsi_change:.2f}, significant: {rsi_significant_change}")
        
        # Confirm bullish divergence
        result = (rsi_decreasing and rsi_still_decreasing and 
                price_not_decreasing and rsi_significant_change)
        
        if result:
            self.logger.debug(f"{symbol}: Bullish divergence confirmed!")
        
        return result
    
    def _check_bearish_divergence(self, symbol: str, data: pd.DataFrame, rsi: pd.Series) -> bool:
        """
        Check for bearish divergence (sell signal).
        
        Args:
            symbol: Trading symbol
            data: Price data
            rsi: RSI values
            
        Returns:
            True if bearish divergence is detected
        """
        rsi_overbought = self.parameters.get("rsi_overbought", 70)
        divergence_lookback = self.parameters.get("divergence_lookback", 5)
        price_change_threshold = self.parameters.get("price_change_threshold", 0.005)
        rsi_change_threshold = self.parameters.get("rsi_change_threshold", 2.0)
        
        # RSI must be in overbought territory
        if rsi.iloc[-1] <= rsi_overbought:
            self.logger.debug(f"{symbol}: RSI {rsi.iloc[-1]:.2f} not in overbought territory (<= {rsi_overbought})")
            return False
        
        # Check if we have enough data
        if len(data) < divergence_lookback + 1 or len(rsi) < divergence_lookback + 1:
            self.logger.debug(f"{symbol}: Not enough data for divergence check")
            return False
        
        # Get past values
        past_prices = data["close"].iloc[-(divergence_lookback+1):-1]
        past_rsi = rsi.iloc[-(divergence_lookback+1):-1]
        
        # Check if RSI is increasing
        rsi_increasing = all(past_rsi.iloc[i] < past_rsi.iloc[i+1] for i in range(len(past_rsi)-1))
        rsi_still_increasing = past_rsi.iloc[-1] < rsi.iloc[-1]
        
        # Check if price is not increasing significantly
        price_change = (data["close"].iloc[-1] - past_prices.iloc[-1]) / past_prices.iloc[-1]
        price_not_increasing = price_change < price_change_threshold
        
        # Check if RSI change is significant
        rsi_change = rsi.iloc[-1] - past_rsi.iloc[0]
        rsi_significant_change = rsi_change > rsi_change_threshold
        
        # Log debug info
        self.logger.debug(f"{symbol}: Bearish check - RSI increasing: {rsi_increasing}, still increasing: {rsi_still_increasing}")
        self.logger.debug(f"{symbol}: Bearish check - Price change: {price_change:.4f}, not increasing: {price_not_increasing}")
        self.logger.debug(f"{symbol}: Bearish check - RSI change: {rsi_change:.2f}, significant: {rsi_significant_change}")
        
        # Confirm bearish divergence
        result = (rsi_increasing and rsi_still_increasing and 
                price_not_increasing and rsi_significant_change)
        
        if result:
            self.logger.debug(f"{symbol}: Bearish divergence confirmed!")
        
        return result