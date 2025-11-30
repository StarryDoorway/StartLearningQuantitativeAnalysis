"""
Momentum strategy implementation.

This module implements a momentum trading strategy based on the idea
that assets that have performed well in the past will continue to perform well.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

from .strategy_base import StrategyBase, Signal, SignalType, SignalStrength
from ..utils.config_loader import get_config
from ..core.event_bus import EventType

logger = logging.getLogger(__name__)


class MomentumStrategy(StrategyBase):
    """
    Momentum trading strategy.
    
    This strategy identifies assets with strong price momentum and trades
    on the expectation that the momentum will continue.
    """
    
    def __init__(self, strategy_id: str, config: Dict[str, Any]):
        """
        Initialize the momentum strategy.
        
        Args:
            strategy_id: Unique strategy identifier
            config: Strategy configuration
        """
        # Default parameters
        default_params = {
            "lookback_period": 20,  # Period for calculating momentum
            "entry_threshold": 0.02,  # Minimum return for entry
            "exit_threshold": -0.01,  # Return threshold for exit
            "position_size": 0.1,  # Position size as fraction of capital
            "min_price": 0.01,  # Minimum price to trade
            "max_price": 1000000,  # Maximum price to trade
            "min_volume": 100,  # Minimum volume to trade
            "use_sma": True,  # Use Simple Moving Average for trend
            "sma_short": 10,  # Short SMA period
            "sma_long": 30,  # Long SMA period
            "use_ema": False,  # Use Exponential Moving Average
            "ema_short": 12,  # Short EMA period
            "ema_long": 26,  # Long EMA period
            "use_macd": True,  # Use MACD for confirmation
            "macd_fast": 12,  # MACD fast EMA period
            "macd_slow": 26,  # MACD slow EMA period
            "macd_signal": 9,  # MACD signal line period
            "use_rsi": False,  # Use RSI for overbought/oversold
            "rsi_period": 14,  # RSI period
            "rsi_overbought": 70,  # RSI overbought level
            "rsi_oversold": 30,  # RSI oversold level
            "use_adx": False,  # Use ADX for trend strength
            "adx_period": 14,  # ADX period
            "adx_threshold": 25,  # ADX threshold for trend
            "confirmation_required": True,  # Require confirmation before entry
            "stop_loss_pct": 0.05,  # Stop loss percentage
            "take_profit_pct": 0.15,  # Take profit percentage
            "max_positions": 5,  # Maximum number of positions
            "correlation_threshold": 0.7,  # Correlation threshold for diversification
            "volatility_adjustment": True,  # Adjust position size based on volatility
            "min_data_points": 50,  # Minimum data points required
            "momentum_lookback": 5,  # Lookback period for momentum calculation
            "momentum_threshold": 0.01  # Minimum momentum for entry
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
        
        self.logger.info(f"Momentum strategy {strategy_id} initialized with parameters: {self.parameters}")
    
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
        
        # Get current price and volume
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
        
        if self.parameters.get("use_sma", True):
            signals.extend(self._generate_sma_signals(symbol, data, indicators, current_position))
        
        if self.parameters.get("use_ema", False):
            signals.extend(self._generate_ema_signals(symbol, data, indicators, current_position))
        
        if self.parameters.get("use_macd", True):
            signals.extend(self._generate_macd_signals(symbol, data, indicators, current_position))
        
        if self.parameters.get("use_rsi", False):
            signals.extend(self._generate_rsi_signals(symbol, data, indicators, current_position))
        
        if self.parameters.get("use_adx", False):
            signals.extend(self._generate_adx_signals(symbol, data, indicators, current_position))
        
        # If not using specific indicators, use simple momentum
        if not (self.parameters.get("use_sma", True) or 
                self.parameters.get("use_ema", False) or 
                self.parameters.get("use_macd", True) or
                self.parameters.get("use_rsi", False) or
                self.parameters.get("use_adx", False)):
            signals.extend(self._generate_simple_momentum_signals(symbol, data, indicators, current_position))
        
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
        
        # Calculate Simple Moving Averages if enabled
        if self.parameters.get("use_sma", True):
            sma_short = self.parameters.get("sma_short", 10)
            sma_long = self.parameters.get("sma_long", 30)
            
            indicators['sma_short'] = data['close'].rolling(window=sma_short).mean()
            indicators['sma_long'] = data['close'].rolling(window=sma_long).mean()
            indicators['sma_cross'] = indicators['sma_short'] > indicators['sma_long']
        
        # Calculate Exponential Moving Averages if enabled
        if self.parameters.get("use_ema", False):
            ema_short = self.parameters.get("ema_short", 12)
            ema_long = self.parameters.get("ema_long", 26)
            
            indicators['ema_short'] = data['close'].ewm(span=ema_short).mean()
            indicators['ema_long'] = data['close'].ewm(span=ema_long).mean()
            indicators['ema_cross'] = indicators['ema_short'] > indicators['ema_long']
        
        # Calculate MACD if enabled
        if self.parameters.get("use_macd", True):
            macd_fast = self.parameters.get("macd_fast", 12)
            macd_slow = self.parameters.get("macd_slow", 26)
            macd_signal = self.parameters.get("macd_signal", 9)
            
            ema_fast = data['close'].ewm(span=macd_fast).mean()
            ema_slow = data['close'].ewm(span=macd_slow).mean()
            
            indicators['macd'] = ema_fast - ema_slow
            indicators['macd_signal'] = indicators['macd'].ewm(span=macd_signal).mean()
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            indicators['macd_cross'] = indicators['macd'] > indicators['macd_signal']
        
        # Calculate RSI if enabled
        if self.parameters.get("use_rsi", False):
            rsi_period = self.parameters.get("rsi_period", 14)
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate ADX if enabled
        if self.parameters.get("use_adx", False):
            adx_period = self.parameters.get("adx_period", 14)
            
            # Calculate True Range
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            tr = np.maximum(high_low, np.maximum(high_close, low_close))
            
            # Calculate Directional Movement
            up_move = data['high'] - data['high'].shift()
            down_move = data['low'].shift() - data['low']
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Calculate ADX
            atr = pd.Series(tr).rolling(window=adx_period).mean()
            plus_di = 100 * (pd.Series(plus_dm).rolling(window=adx_period).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(window=adx_period).mean() / atr)
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            indicators['adx'] = pd.Series(dx).rolling(window=adx_period).mean()
        
        # Calculate simple momentum
        lookback = self.parameters.get("momentum_lookback", 5)
        indicators['momentum'] = data['close'].pct_change(lookback)
        
        # Calculate volatility for position sizing
        volatility_period = self.parameters.get("lookback_period", 20)
        indicators['volatility'] = data['close'].pct_change().rolling(window=volatility_period).std()
    
    def _generate_sma_signals(self, symbol: str, data: pd.DataFrame, indicators: Dict[str, pd.Series], current_position: float) -> List[Signal]:
        """
        Generate signals using Simple Moving Averages.
        
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
        sma_short = indicators['sma_short'].iloc[-1]
        sma_long = indicators['sma_long'].iloc[-1]
        sma_cross = indicators['sma_cross'].iloc[-1]
        sma_cross_prev = indicators['sma_cross'].iloc[-2] if len(indicators['sma_cross']) > 1 else False
        
        # Entry signals
        if current_position == 0:  # No position
            # Short SMA crosses above long SMA - potential buy signal
            if sma_cross and not sma_cross_prev:
                # Check momentum
                momentum = indicators['momentum'].iloc[-1]
                momentum_threshold = self.parameters.get("momentum_threshold", 0.01)
                
                if momentum > momentum_threshold:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        strength=SignalStrength.MODERATE,
                        price=current_price,
                        confidence=min(1.0, momentum * 10),  # Higher confidence for stronger momentum
                        timestamp=datetime.now(),
                        metadata={
                            "strategy": "momentum",
                            "indicator": "sma_cross",
                            "sma_short": sma_short,
                            "sma_long": sma_long,
                            "momentum": momentum
                        }
                    )
                    signals.append(signal)
            
            # Short SMA crosses below long SMA - potential sell signal
            elif not sma_cross and sma_cross_prev:
                # Check momentum
                momentum = indicators['momentum'].iloc[-1]
                momentum_threshold = self.parameters.get("momentum_threshold", 0.01)
                
                if momentum < -momentum_threshold:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        strength=SignalStrength.MODERATE,
                        price=current_price,
                        confidence=min(1.0, abs(momentum) * 10),  # Higher confidence for stronger momentum
                        timestamp=datetime.now(),
                        metadata={
                            "strategy": "momentum",
                            "indicator": "sma_cross",
                            "sma_short": sma_short,
                            "sma_long": sma_long,
                            "momentum": momentum
                        }
                    )
                    signals.append(signal)
        
        # Exit signals
        elif current_position > 0:  # Long position
            # Short SMA crosses below long SMA - exit signal
            if not sma_cross and sma_cross_prev:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.WEAK,
                    price=current_price,
                    confidence=0.7,
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "momentum",
                        "indicator": "sma_cross",
                        "action": "exit_long"
                    }
                )
                signals.append(signal)
        
        elif current_position < 0:  # Short position
            # Short SMA crosses above long SMA - exit signal
            if sma_cross and not sma_cross_prev:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.WEAK,
                    price=current_price,
                    confidence=0.7,
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "momentum",
                        "indicator": "sma_cross",
                        "action": "exit_short"
                    }
                )
                signals.append(signal)
        
        return signals
    
    def _generate_ema_signals(self, symbol: str, data: pd.DataFrame, indicators: Dict[str, pd.Series], current_position: float) -> List[Signal]:
        """
        Generate signals using Exponential Moving Averages.
        
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
        ema_short = indicators['ema_short'].iloc[-1]
        ema_long = indicators['ema_long'].iloc[-1]
        ema_cross = indicators['ema_cross'].iloc[-1]
        ema_cross_prev = indicators['ema_cross'].iloc[-2] if len(indicators['ema_cross']) > 1 else False
        
        # Entry signals
        if current_position == 0:  # No position
            # Short EMA crosses above long EMA - potential buy signal
            if ema_cross and not ema_cross_prev:
                # Check momentum
                momentum = indicators['momentum'].iloc[-1]
                momentum_threshold = self.parameters.get("momentum_threshold", 0.01)
                
                if momentum > momentum_threshold:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        strength=SignalStrength.MODERATE,
                        price=current_price,
                        confidence=min(1.0, momentum * 10),  # Higher confidence for stronger momentum
                        timestamp=datetime.now(),
                        metadata={
                            "strategy": "momentum",
                            "indicator": "ema_cross",
                            "ema_short": ema_short,
                            "ema_long": ema_long,
                            "momentum": momentum
                        }
                    )
                    signals.append(signal)
            
            # Short EMA crosses below long EMA - potential sell signal
            elif not ema_cross and ema_cross_prev:
                # Check momentum
                momentum = indicators['momentum'].iloc[-1]
                momentum_threshold = self.parameters.get("momentum_threshold", 0.01)
                
                if momentum < -momentum_threshold:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        strength=SignalStrength.MODERATE,
                        price=current_price,
                        confidence=min(1.0, abs(momentum) * 10),  # Higher confidence for stronger momentum
                        timestamp=datetime.now(),
                        metadata={
                            "strategy": "momentum",
                            "indicator": "ema_cross",
                            "ema_short": ema_short,
                            "ema_long": ema_long,
                            "momentum": momentum
                        }
                    )
                    signals.append(signal)
        
        # Exit signals
        elif current_position > 0:  # Long position
            # Short EMA crosses below long EMA - exit signal
            if not ema_cross and ema_cross_prev:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.WEAK,
                    price=current_price,
                    confidence=0.7,
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "momentum",
                        "indicator": "ema_cross",
                        "action": "exit_long"
                    }
                )
                signals.append(signal)
        
        elif current_position < 0:  # Short position
            # Short EMA crosses above long EMA - exit signal
            if ema_cross and not ema_cross_prev:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.WEAK,
                    price=current_price,
                    confidence=0.7,
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "momentum",
                        "indicator": "ema_cross",
                        "action": "exit_short"
                    }
                )
                signals.append(signal)
        
        return signals
    
    def _generate_macd_signals(self, symbol: str, data: pd.DataFrame, indicators: Dict[str, pd.Series], current_position: float) -> List[Signal]:
        """
        Generate signals using MACD.
        
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
        macd = indicators['macd'].iloc[-1]
        macd_signal = indicators['macd_signal'].iloc[-1]
        macd_histogram = indicators['macd_histogram'].iloc[-1]
        macd_cross = indicators['macd_cross'].iloc[-1]
        macd_cross_prev = indicators['macd_cross'].iloc[-2] if len(indicators['macd_cross']) > 1 else False
        
        # Entry signals
        if current_position == 0:  # No position
            # MACD crosses above signal line - potential buy signal
            if macd_cross and not macd_cross_prev:
                # Check momentum
                momentum = indicators['momentum'].iloc[-1]
                momentum_threshold = self.parameters.get("momentum_threshold", 0.01)
                
                if momentum > momentum_threshold:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        strength=SignalStrength.MODERATE,
                        price=current_price,
                        confidence=min(1.0, momentum * 10),  # Higher confidence for stronger momentum
                        timestamp=datetime.now(),
                        metadata={
                            "strategy": "momentum",
                            "indicator": "macd_cross",
                            "macd": macd,
                            "macd_signal": macd_signal,
                            "macd_histogram": macd_histogram,
                            "momentum": momentum
                        }
                    )
                    signals.append(signal)
            
            # MACD crosses below signal line - potential sell signal
            elif not macd_cross and macd_cross_prev:
                # Check momentum
                momentum = indicators['momentum'].iloc[-1]
                momentum_threshold = self.parameters.get("momentum_threshold", 0.01)
                
                if momentum < -momentum_threshold:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        strength=SignalStrength.MODERATE,
                        price=current_price,
                        confidence=min(1.0, abs(momentum) * 10),  # Higher confidence for stronger momentum
                        timestamp=datetime.now(),
                        metadata={
                            "strategy": "momentum",
                            "indicator": "macd_cross",
                            "macd": macd,
                            "macd_signal": macd_signal,
                            "macd_histogram": macd_histogram,
                            "momentum": momentum
                        }
                    )
                    signals.append(signal)
        
        # Exit signals
        elif current_position > 0:  # Long position
            # MACD crosses below signal line - exit signal
            if not macd_cross and macd_cross_prev:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.WEAK,
                    price=current_price,
                    confidence=0.7,
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "momentum",
                        "indicator": "macd_cross",
                        "action": "exit_long"
                    }
                )
                signals.append(signal)
        
        elif current_position < 0:  # Short position
            # MACD crosses above signal line - exit signal
            if macd_cross and not macd_cross_prev:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.WEAK,
                    price=current_price,
                    confidence=0.7,
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "momentum",
                        "indicator": "macd_cross",
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
            # RSI crosses above 50 - potential buy signal
            if rsi > 50:
                # Check momentum
                momentum = indicators['momentum'].iloc[-1]
                momentum_threshold = self.parameters.get("momentum_threshold", 0.01)
                
                if momentum > momentum_threshold:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        strength=SignalStrength.WEAK,
                        price=current_price,
                        confidence=min(1.0, momentum * 10),  # Higher confidence for stronger momentum
                        timestamp=datetime.now(),
                        metadata={
                            "strategy": "momentum",
                            "indicator": "rsi",
                            "rsi": rsi,
                            "momentum": momentum
                        }
                    )
                    signals.append(signal)
            
            # RSI crosses below 50 - potential sell signal
            elif rsi < 50:
                # Check momentum
                momentum = indicators['momentum'].iloc[-1]
                momentum_threshold = self.parameters.get("momentum_threshold", 0.01)
                
                if momentum < -momentum_threshold:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        strength=SignalStrength.WEAK,
                        price=current_price,
                        confidence=min(1.0, abs(momentum) * 10),  # Higher confidence for stronger momentum
                        timestamp=datetime.now(),
                        metadata={
                            "strategy": "momentum",
                            "indicator": "rsi",
                            "rsi": rsi,
                            "momentum": momentum
                        }
                    )
                    signals.append(signal)
        
        # Exit signals
        elif current_position > 0:  # Long position
            # RSI crosses above overbought level - exit signal
            if rsi > rsi_overbought:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.WEAK,
                    price=current_price,
                    confidence=0.6,
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "momentum",
                        "indicator": "rsi",
                        "action": "exit_long",
                        "rsi": rsi
                    }
                )
                signals.append(signal)
        
        elif current_position < 0:  # Short position
            # RSI crosses below oversold level - exit signal
            if rsi < rsi_oversold:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.WEAK,
                    price=current_price,
                    confidence=0.6,
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "momentum",
                        "indicator": "rsi",
                        "action": "exit_short",
                        "rsi": rsi
                    }
                )
                signals.append(signal)
        
        return signals
    
    def _generate_adx_signals(self, symbol: str, data: pd.DataFrame, indicators: Dict[str, pd.Series], current_position: float) -> List[Signal]:
        """
        Generate signals using ADX.
        
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
        adx = indicators['adx'].iloc[-1]
        adx_threshold = self.parameters.get("adx_threshold", 25)
        
        # ADX is used for confirmation, not for direct signals
        # We'll use it to confirm momentum signals
        if adx < adx_threshold:
            # Trend is weak, don't generate signals
            return signals
        
        # Check momentum
        momentum = indicators['momentum'].iloc[-1]
        momentum_threshold = self.parameters.get("momentum_threshold", 0.01)
        
        # Entry signals
        if current_position == 0:  # No position
            # Strong upward momentum - buy signal
            if momentum > momentum_threshold:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.MODERATE,
                    price=current_price,
                    confidence=min(1.0, momentum * 10 * (adx / adx_threshold)),  # Higher confidence for stronger ADX
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "momentum",
                        "indicator": "adx",
                        "adx": adx,
                        "momentum": momentum
                    }
                )
                signals.append(signal)
            
            # Strong downward momentum - sell signal
            elif momentum < -momentum_threshold:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.MODERATE,
                    price=current_price,
                    confidence=min(1.0, abs(momentum) * 10 * (adx / adx_threshold)),  # Higher confidence for stronger ADX
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "momentum",
                        "indicator": "adx",
                        "adx": adx,
                        "momentum": momentum
                    }
                )
                signals.append(signal)
        
        return signals
    
    def _generate_simple_momentum_signals(self, symbol: str, data: pd.DataFrame, indicators: Dict[str, pd.Series], current_position: float) -> List[Signal]:
        """
        Generate signals using simple momentum.
        
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
        momentum = indicators['momentum'].iloc[-1]
        entry_threshold = self.parameters.get("entry_threshold", 0.02)
        exit_threshold = self.parameters.get("exit_threshold", -0.01)
        
        # Entry signals
        if current_position == 0:  # No position
            # Strong upward momentum - buy signal
            if momentum > entry_threshold:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.MODERATE,
                    price=current_price,
                    confidence=min(1.0, momentum / entry_threshold),  # Higher confidence for stronger momentum
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "momentum",
                        "indicator": "simple_momentum",
                        "momentum": momentum
                    }
                )
                signals.append(signal)
            
            # Strong downward momentum - sell signal
            elif momentum < -entry_threshold:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.MODERATE,
                    price=current_price,
                    confidence=min(1.0, abs(momentum) / entry_threshold),  # Higher confidence for stronger momentum
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "momentum",
                        "indicator": "simple_momentum",
                        "momentum": momentum
                    }
                )
                signals.append(signal)
        
        # Exit signals
        elif current_position > 0:  # Long position
            # Momentum reverses - exit signal
            if momentum < exit_threshold:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.WEAK,
                    price=current_price,
                    confidence=0.7,
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "momentum",
                        "indicator": "simple_momentum",
                        "action": "exit_long",
                        "momentum": momentum
                    }
                )
                signals.append(signal)
        
        elif current_position < 0:  # Short position
            # Momentum reverses - exit signal
            if momentum > -exit_threshold:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.WEAK,
                    price=current_price,
                    confidence=0.7,
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "momentum",
                        "indicator": "simple_momentum",
                        "action": "exit_short",
                        "momentum": momentum
                    }
                )
                signals.append(signal)
        
        return signals
    
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
                take_profit_pct = self.parameters.get("take_profit_pct", 0.15)
                
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