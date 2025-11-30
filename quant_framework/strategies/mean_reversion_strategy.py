"""
Mean reversion strategy implementation.

This module implements a mean reversion trading strategy based on the idea
that asset prices tend to revert to their historical mean over time.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

from .strategy_base import StrategyBase, Signal, SignalType, SignalStrength
from ..core.event_bus import EventType, Event, get_event_bus
from ..utils.config_loader import get_config
from ..managers.order_manager import OrderManager

logger = logging.getLogger(__name__)


class MeanReversionStrategy(StrategyBase):
    """
    Mean reversion trading strategy.
    
    This strategy identifies assets that have deviated significantly from their
    historical mean and trades on the expectation that they will revert.
    """
    
    def __init__(self, strategy_id: str, config: Dict[str, Any]):
        """
        Initialize the mean reversion strategy.
        
        Args:
            strategy_id: Unique strategy identifier
            config: Strategy configuration
        """
        # Default parameters
        default_params = {
            "lookback_period": 20,  # Period for calculating mean and std
            "entry_threshold": 2.0,  # Number of std devs for entry
            "exit_threshold": 0.5,  # Number of std devs for exit
            "position_size": 0.1,  # Position size as fraction of capital
            "min_price": 0.01,  # Minimum price to trade
            "max_price": 1000000,  # Maximum price to trade
            "min_volume": 100,  # Minimum volume to trade
            "use_exponential": False,  # Use exponential moving average
            "use_bollinger_bands": True,  # Use Bollinger Bands for signals
            "bb_period": 20,  # Bollinger Bands period
            "bb_std": 2.0,  # Bollinger Bands standard deviation
            "rsi_period": 14,  # RSI period
            "rsi_overbought": 70,  # RSI overbought level
            "rsi_oversold": 30,  # RSI oversold level
            "use_rsi": False,  # Use RSI for confirmation
            "confirmation_required": True,  # Require confirmation before entry
            "stop_loss_pct": 0.05,  # Stop loss percentage
            "take_profit_pct": 0.1,  # Take profit percentage
            "max_positions": 5,  # Maximum number of positions
            "correlation_threshold": 0.7,  # Correlation threshold for diversification
            "volatility_adjustment": True,  # Adjust position size based on volatility
            "min_data_points": 50  # Minimum data points required
        }
        
        # Merge with provided parameters
        if "parameters" in config:
            default_params.update(config["parameters"])
        config["parameters"] = default_params
        
        # Initialize base class
        super().__init__(strategy_id, config)
        
        # Additional state variables
        self.entry_prices: Dict[str, float] = {}
        self.stop_losses: Dict[str, float] = {}
        self.take_profits: Dict[str, float] = {}
        self.last_signals: Dict[str, Signal] = {}
        
        # Technical indicators cache
        self.indicators: Dict[str, Dict[str, pd.Series]] = {}
        
        self.logger.info(f"Mean reversion strategy {strategy_id} initialized with parameters: {self.parameters}")
    
    def _initialize(self) -> None:
        """Initialize strategy-specific components."""
        # Subscribe to market data events
        self.event_bus.subscribe(event_type=EventType.MARKET_DATA, callback=self.on_bar)
        
        # Subscribe to portfolio events
        self.event_bus.subscribe(event_type=EventType.PORTFOLIO_UPDATE, callback=self._on_portfolio_update)
    
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
        # Check if we have enough data
        min_data_points = self.parameters.get("min_data_points", 50)
        if len(data) < min_data_points:
            self.logger.warning(f"Not enough data for {symbol}: {len(data)} < {min_data_points}")
            return []
        
        # Calculate indicators
        self._calculate_indicators(symbol, data)
        
        # Get latest indicators
        indicators = self.indicators.get(symbol, {})
        if not indicators:
            return []
        
        # Get current price
        current_price = data['close'].iloc[-1]
        current_volume = data['volume'].iloc[-1]
        
        # Check price and volume constraints
        min_price = self.parameters.get("min_price", 0.01)
        max_price = self.parameters.get("max_price", 1000000)
        min_volume = self.parameters.get("min_volume", 100)
        
        if not (min_price <= current_price <= max_price) or current_volume < min_volume:
            return []
        
        # Check if we already have a position
        current_position = self.current_positions.get(symbol, 0)
        
        # Generate signals
        signals = []
        
        if self.parameters.get("use_bollinger_bands", True):
            signals.extend(self._generate_bollinger_band_signals(symbol, data, indicators, current_position))
        
        if self.parameters.get("use_rsi", False):
            signals.extend(self._generate_rsi_signals(symbol, data, indicators, current_position))
        
        # If not using specific indicators, use simple mean reversion
        if not self.parameters.get("use_bollinger_bands", True) and not self.parameters.get("use_rsi", False):
            signals.extend(self._generate_simple_mean_reversion_signals(symbol, data, indicators, current_position))
        
        # Apply position limits
        signals = self._apply_position_limits(signals)
        
        # Update last signals
        for signal in signals:
            self.last_signals[symbol] = signal
        
        return signals
    
    def _calculate_indicators(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Calculate technical indicators for a symbol.
        
        Args:
            symbol: Trading symbol
            data: Historical market data
        """
        # Initialize indicators dict if needed
        if symbol not in self.indicators:
            self.indicators[symbol] = {}
        
        indicators = self.indicators[symbol]
        
        # Calculate moving average
        lookback = self.parameters.get("lookback_period", 20)
        use_exponential = self.parameters.get("use_exponential", False)
        
        if use_exponential:
            indicators['mean'] = data['close'].ewm(span=lookback).mean()
        else:
            indicators['mean'] = data['close'].rolling(window=lookback).mean()
        
        # Calculate standard deviation
        indicators['std'] = data['close'].rolling(window=lookback).std()
        
        # Calculate z-score
        indicators['zscore'] = (data['close'] - indicators['mean']) / indicators['std']
        
        # Calculate Bollinger Bands if enabled
        if self.parameters.get("use_bollinger_bands", True):
            bb_period = self.parameters.get("bb_period", 20)
            bb_std = self.parameters.get("bb_std", 2.0)
            
            indicators['bb_middle'] = data['close'].rolling(window=bb_period).mean()
            indicators['bb_std'] = data['close'].rolling(window=bb_period).std()
            indicators['bb_upper'] = indicators['bb_middle'] + (indicators['bb_std'] * bb_std)
            indicators['bb_lower'] = indicators['bb_middle'] - (indicators['bb_std'] * bb_std)
            indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle']
            indicators['bb_position'] = (data['close'] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        # Calculate RSI if enabled
        if self.parameters.get("use_rsi", False):
            rsi_period = self.parameters.get("rsi_period", 14)
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate volatility for position sizing
        indicators['volatility'] = data['close'].pct_change().rolling(window=lookback).std()
    
    def _generate_bollinger_band_signals(self, symbol: str, data: pd.DataFrame, indicators: Dict[str, pd.Series], current_position: float) -> List[Signal]:
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
        
        # Get latest values
        current_price = data['close'].iloc[-1]
        bb_upper = indicators['bb_upper'].iloc[-1]
        bb_lower = indicators['bb_lower'].iloc[-1]
        bb_position = indicators['bb_position'].iloc[-1]
        bb_width = indicators['bb_width'].iloc[-1]
        
        # Entry signals
        if current_position == 0:  # No position
            # Price below lower band - potential buy signal
            if current_price < bb_lower:
                # Check if we need confirmation
                if not self.parameters.get("confirmation_required", True) or self._is_band_widening(symbol, indicators):
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        strength=SignalStrength.MODERATE,
                        price=current_price,
                        confidence=min(1.0, bb_width * 5),  # Higher confidence for wider bands
                        timestamp=datetime.now(),
                        metadata={
                            "strategy": "mean_reversion",
                            "indicator": "bollinger_bands",
                            "bb_position": bb_position,
                            "bb_width": bb_width
                        }
                    )
                    signals.append(signal)
            
            # Price above upper band - potential sell signal
            elif current_price > bb_upper:
                # Check if we need confirmation
                if not self.parameters.get("confirmation_required", True) or self._is_band_widening(symbol, indicators):
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        strength=SignalStrength.MODERATE,
                        price=current_price,
                        confidence=min(1.0, bb_width * 5),  # Higher confidence for wider bands
                        timestamp=datetime.now(),
                        metadata={
                            "strategy": "mean_reversion",
                            "indicator": "bollinger_bands",
                            "bb_position": bb_position,
                            "bb_width": bb_width
                        }
                    )
                    signals.append(signal)
        
        # Exit signals
        elif current_position > 0:  # Long position
            # Price crosses back above middle band - exit signal
            if current_price > indicators['bb_middle'].iloc[-1]:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.WEAK,
                    price=current_price,
                    confidence=0.7,
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
            if current_price < indicators['bb_middle'].iloc[-1]:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.WEAK,
                    price=current_price,
                    confidence=0.7,
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "mean_reversion",
                        "indicator": "bollinger_bands",
                        "action": "exit_short"
                    }
                )
                signals.append(signal)
        
        return signals
    
    def _generate_rsi_signals(self, symbol: str, data: pd.DataFrame, indicators: Dict[str, pd.Series], current_position: float) -> List[Signal]:
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
        
        # Get latest values
        current_price = data['close'].iloc[-1]
        rsi = indicators['rsi'].iloc[-1]
        rsi_overbought = self.parameters.get("rsi_overbought", 70)
        rsi_oversold = self.parameters.get("rsi_oversold", 30)
        
        # Entry signals
        if current_position == 0:  # No position
            # RSI below oversold level - potential buy signal
            if rsi < rsi_oversold:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.WEAK,
                    price=current_price,
                    confidence=1.0 - (rsi / rsi_oversold),  # Higher confidence for lower RSI
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "mean_reversion",
                        "indicator": "rsi",
                        "rsi": rsi
                    }
                )
                signals.append(signal)
            
            # RSI above overbought level - potential sell signal
            elif rsi > rsi_overbought:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.WEAK,
                    price=current_price,
                    confidence=(rsi - rsi_overbought) / (100 - rsi_overbought),  # Higher confidence for higher RSI
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
    
    def _generate_simple_mean_reversion_signals(self, symbol: str, data: pd.DataFrame, indicators: Dict[str, pd.Series], current_position: float) -> List[Signal]:
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
    
    def _is_band_widening(self, symbol: str, indicators: Dict[str, pd.Series]) -> bool:
        """
        Check if Bollinger Bands are widening.
        
        Args:
            symbol: Trading symbol
            indicators: Technical indicators
            
        Returns:
            True if bands are widening
        """
        if 'bb_width' not in indicators or len(indicators['bb_width']) < 2:
            return False
        
        # Check if current width is greater than previous width
        current_width = indicators['bb_width'].iloc[-1]
        previous_width = indicators['bb_width'].iloc[-2]
        
        return current_width > previous_width
    
    def _apply_position_limits(self, signals: List[Signal]) -> List[Signal]:
        """
        Apply position limits to signals.
        
        Args:
            signals: List of signals
            
        Returns:
            Filtered list of signals
        """
        # Check maximum number of positions
        max_positions = self.parameters.get("max_positions", 5)
        current_position_count = len([pos for pos in self.current_positions.values() if pos != 0])
        
        if current_position_count >= max_positions:
            # Only allow exit signals
            return [signal for signal in signals if 
                   (signal.signal_type == SignalType.BUY and self.current_positions.get(signal.symbol, 0) < 0) or
                   (signal.signal_type == SignalType.SELL and self.current_positions.get(signal.symbol, 0) > 0)]
        
        # Apply volatility adjustment to position size
        if self.parameters.get("volatility_adjustment", True):
            for signal in signals:
                symbol = signal.symbol
                if symbol in self.indicators and 'volatility' in self.indicators[symbol]:
                    volatility = self.indicators[symbol]['volatility'].iloc[-1]
                    # Inverse relationship between volatility and position size
                    adjusted_size = self.parameters.get("position_size", 0.1) / (1 + volatility * 10)
                    signal.quantity = adjusted_size
        
        return signals
    
    def on_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Called when a trade is executed.
        
        Args:
            trade_data: Trade execution data
        """
        # Call parent method
        super().on_trade(trade_data)
        
        # Update entry prices, stop losses, and take profits
        symbol = trade_data.get("symbol")
        if symbol:
            if trade_data.get("side") == "buy":
                self.entry_prices[symbol] = trade_data.get("price", 0)
                # Set stop loss and take profit
                entry_price = self.entry_prices[symbol]
                stop_loss_pct = self.parameters.get("stop_loss_pct", 0.05)
                take_profit_pct = self.parameters.get("take_profit_pct", 0.1)
                
                self.stop_losses[symbol] = entry_price * (1 - stop_loss_pct)
                self.take_profits[symbol] = entry_price * (1 + take_profit_pct)
            elif trade_data.get("side") == "sell":
                # Clear position-related data
                if symbol in self.entry_prices:
                    del self.entry_prices[symbol]
                if symbol in self.stop_losses:
                    del self.stop_losses[symbol]
                if symbol in self.take_profits:
                    del self.take_profits[symbol]