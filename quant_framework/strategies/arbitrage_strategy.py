"""
Arbitrage strategy implementation.

This module implements various arbitrage strategies that exploit price
inefficiencies across different markets or instruments.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict

from quant_framework.core.event_bus import get_event_bus, EventType, Event
from quant_framework.strategies.strategy_base import StrategyBase
from quant_framework.strategies.base.signal_types import Signal, SignalType, SignalStrength

logger = logging.getLogger(__name__)


class ArbitrageStrategy(StrategyBase):
    """
    Arbitrage trading strategy.
    
    This strategy identifies and exploits price inefficiencies across
    different markets or instruments.
    """
    
    def __init__(self, strategy_id: str, config: Dict[str, Any]):
        """
        Initialize the arbitrage strategy.
        
        Args:
            strategy_id: Unique strategy identifier
            config: Strategy configuration
        """
        # Default parameters
        default_params = {
            "arbitrage_type": "statistical",  # Type of arbitrage: statistical, triangular, cross_exchange
            "lookback_period": 20,  # Period for calculating statistics
            "entry_threshold": 2.0,  # Standard deviations for entry
            "exit_threshold": 0.5,  # Standard deviations for exit
            "position_size": 0.1,  # Position size as fraction of capital
            "min_price": 0.01,  # Minimum price to trade
            "max_price": 1000000,  # Maximum price to trade
            "min_volume": 100,  # Minimum volume to trade
            "cointegration_period": 60,  # Period for cointegration test
            "half_life_period": 20,  # Period for half-life calculation
            "zscore_period": 20,  # Period for z-score calculation
            "pairs": [],  # List of trading pairs for pairs trading
            "triangular_sets": [],  # List of triangular arbitrage sets
            "clients": [],  # List of clients for cross-exchange arbitrage
            "fee_rate": 0.001,  # Trading fee rate
            "min_profit_threshold": 0.002,  # Minimum profit threshold
            "max_positions": 5,  # Maximum number of positions
            "correlation_threshold": 0.8,  # Correlation threshold for pairs
            "min_data_points": 100,  # Minimum data points required
            "rebalance_frequency": 1,  # Rebalance frequency in days
            "use_hedge_ratio": True,  # Use hedge ratio for pairs trading
            "use_dynamic_thresholds": True,  # Use dynamic thresholds
            "risk_adjusted_position": True,  # Adjust position size based on risk
            "max_drawdown": 0.1,  # Maximum drawdown
            "volatility_lookback": 20,  # Period for volatility calculation
            "spread_lookback": 20,  # Period for spread calculation
            "use_stop_loss": True,  # Use stop loss
            "stop_loss_pct": 0.05,  # Stop loss percentage
            "use_take_profit": True,  # Use take profit
            "take_profit_pct": 0.15  # Take profit percentage
        }
        
        # Merge with provided parameters
        if "parameters" in config:
            default_params.update(config["parameters"])
        config["parameters"] = default_params
        
        # Initialize base class
        super().__init__(strategy_id, config)
        
        # Additional state variables
        self.spreads: Dict[str, pd.Series] = {}
        self.hedge_ratios: Dict[str, float] = {}
        self.zscores: Dict[str, pd.Series] = {}
        self.entry_prices: Dict[str, Dict[str, float]] = {}
        self.last_signals: Dict[str, Signal] = {}
        self.pair_data: Dict[str, Dict[str, pd.DataFrame]] = defaultdict(dict)
        self.cointegration_results: Dict[str, Dict[str, Any]] = {}
        self.half_lives: Dict[str, float] = {}
        self.correlations: Dict[str, float] = {}
        self.volatilities: Dict[str, float] = {}
        
        self.logger.info(f"Arbitrage strategy {strategy_id} initialized with parameters: {self.parameters}")
    
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
        min_data_points = self.parameters.get("min_data_points", 100)
        if len(data) < min_data_points:
            self.logger.warning(f"Not enough data for {symbol}: {len(data)} < {min_data_points}")
            return []
        
        # Store data for pairs analysis
        self.pair_data[symbol] = data
        
        # Calculate signals based on arbitrage type
        arbitrage_type = self.parameters.get("arbitrage_type", "statistical")
        
        if arbitrage_type == "statistical":
            return self._calculate_statistical_arbitrage_signals(symbol, data)
        elif arbitrage_type == "triangular":
            return self._calculate_triangular_arbitrage_signals(symbol, data)
        elif arbitrage_type == "cross_exchange":
            return self._calculate_cross_exchange_arbitrage_signals(symbol, data)
        else:
            self.logger.warning(f"Unknown arbitrage type: {arbitrage_type}")
            return []
    
    def _calculate_statistical_arbitrage_signals(self, symbol: str, data: pd.DataFrame) -> List[Signal]:
        """
        Calculate signals for statistical arbitrage (pairs trading).
        
        Args:
            symbol: Trading symbol
            data: Historical market data
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Get pairs for this symbol
        pairs = self.parameters.get("pairs", [])
        symbol_pairs = [pair for pair in pairs if symbol in pair]
        
        for pair in symbol_pairs:
            # Get the other symbol in the pair
            other_symbol = pair[0] if pair[1] == symbol else pair[1]
            
            # Check if we have data for the other symbol
            if other_symbol not in self.pair_data:
                continue
            
            # Get data for both symbols
            symbol_data = data
            other_data = self.pair_data[other_symbol]
            
            # Align data by timestamp
            aligned_data = pd.concat([symbol_data['close'], other_data['close']], axis=1, join='inner')
            aligned_data.columns = [symbol, other_symbol]
            
            # Check if we have enough aligned data
            min_data_points = self.parameters.get("min_data_points", 100)
            if len(aligned_data) < min_data_points:
                continue
            
            # Calculate spread and hedge ratio
            pair_key = f"{symbol}_{other_symbol}"
            self._calculate_spread_and_hedge_ratio(pair_key, aligned_data)
            
            # Calculate z-score
            self._calculate_zscore(pair_key)
            
            # Generate signals based on z-score
            pair_signals = self._generate_pair_signals(pair_key, symbol, other_symbol)
            signals.extend(pair_signals)
        
        return signals
    
    def _calculate_triangular_arbitrage_signals(self, symbol: str, data: pd.DataFrame) -> List[Signal]:
        """
        Calculate signals for triangular arbitrage.
        
        Args:
            symbol: Trading symbol
            data: Historical market data
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Get triangular sets that include this symbol
        triangular_sets = self.parameters.get("triangular_sets", [])
        symbol_sets = [tri_set for tri_set in triangular_sets if symbol in tri_set]
        
        for tri_set in symbol_sets:
            # Get the other two symbols in the set
            other_symbols = [s for s in tri_set if s != symbol]
            if len(other_symbols) != 2:
                continue
            
            symbol1, symbol2 = other_symbols
            
            # Check if we have data for the other symbols
            if symbol1 not in self.pair_data or symbol2 not in self.pair_data:
                continue
            
            # Get data for all three symbols
            symbol_data = data
            symbol1_data = self.pair_data[symbol1]
            symbol2_data = self.pair_data[symbol2]
            
            # Align data by timestamp
            aligned_data = pd.concat([
                symbol_data['close'], 
                symbol1_data['close'], 
                symbol2_data['close']
            ], axis=1, join='inner')
            aligned_data.columns = [symbol, symbol1, symbol2]
            
            # Check if we have enough aligned data
            min_data_points = self.parameters.get("min_data_points", 100)
            if len(aligned_data) < min_data_points:
                continue
            
            # Calculate triangular arbitrage opportunity
            tri_signals = self._generate_triangular_signals(tri_set, aligned_data)
            signals.extend(tri_signals)
        
        return signals
    
    def _calculate_cross_exchange_arbitrage_signals(self, symbol: str, data: pd.DataFrame) -> List[Signal]:
        """
        Calculate signals for cross-exchange arbitrage.
        
        Args:
            symbol: Trading symbol
            data: Historical market data
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # For cross-exchange arbitrage, we would need data from multiple clients
        # This is a simplified implementation
        exchanges = self.parameters.get("clients", [])
        if len(exchanges) < 2:
            return signals
        
        # In a real implementation, we would have data from different clients
        # For now, we'll simulate with a simple price difference check
        current_price = data['close'].iloc[-1]
        
        # Simulate price differences (in a real implementation, this would come from different clients)
        price_diff = np.random.normal(0, 0.001)  # Random price difference
        other_price = current_price * (1 + price_diff)
        
        # Calculate potential profit after fees
        fee_rate = self.parameters.get("fee_rate", 0.001)
        min_profit_threshold = self.parameters.get("min_profit_threshold", 0.002)
        
        # Buy on the cheaper exchange, sell on the more expensive one
        if current_price < other_price:
            # Buy on current exchange, sell on other
            profit_pct = (other_price - current_price) / current_price - 2 * fee_rate
            if profit_pct > min_profit_threshold:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.STRONG,
                    price=current_price,
                    confidence=min(1.0, profit_pct / min_profit_threshold),
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "arbitrage",
                        "type": "cross_exchange",
                        "buy_exchange": "current",
                        "sell_exchange": "other",
                        "buy_price": current_price,
                        "sell_price": other_price,
                        "profit_pct": profit_pct
                    }
                )
                signals.append(signal)
        else:
            # Buy on other exchange, sell on current
            profit_pct = (current_price - other_price) / other_price - 2 * fee_rate
            if profit_pct > min_profit_threshold:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.STRONG,
                    price=current_price,
                    confidence=min(1.0, profit_pct / min_profit_threshold),
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "arbitrage",
                        "type": "cross_exchange",
                        "buy_exchange": "other",
                        "sell_exchange": "current",
                        "buy_price": other_price,
                        "sell_price": current_price,
                        "profit_pct": profit_pct
                    }
                )
                signals.append(signal)
        
        return signals
    
    def _calculate_spread_and_hedge_ratio(self, pair_key: str, data: pd.DataFrame) -> None:
        """
        Calculate spread and hedge ratio for a pair.
        
        Args:
            pair_key: Key for the pair (symbol1_symbol2)
            data: Aligned price data for the pair
        """
        symbols = pair_key.split('_')
        symbol1, symbol2 = symbols[0], symbols[1]
        
        # Get price series
        y = data[symbol1]
        x = data[symbol2]
        
        # Calculate hedge ratio using linear regression
        # Add constant for intercept
        x_with_const = np.column_stack([np.ones(len(x)), x])
        
        # Calculate regression coefficients
        coeffs = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
        hedge_ratio = coeffs[1]
        
        # Store hedge ratio
        self.hedge_ratios[pair_key] = hedge_ratio
        
        # Calculate spread
        spread = y - hedge_ratio * x
        self.spreads[pair_key] = spread
        
        # Calculate correlation
        correlation = np.corrcoef(y, x)[0, 1]
        self.correlations[pair_key] = correlation
        
        # Calculate volatilities
        volatility_period = self.parameters.get("volatility_lookback", 20)
        symbol1_vol = y.pct_change().rolling(window=volatility_period).std().iloc[-1]
        symbol2_vol = x.pct_change().rolling(window=volatility_period).std().iloc[-1]
        self.volatilities[f"{pair_key}_symbol1"] = symbol1_vol
        self.volatilities[f"{pair_key}_symbol2"] = symbol2_vol
    
    def _calculate_zscore(self, pair_key: str) -> None:
        """
        Calculate z-score for a pair's spread.
        
        Args:
            pair_key: Key for the pair
        """
        if pair_key not in self.spreads:
            return
        
        spread = self.spreads[pair_key]
        zscore_period = self.parameters.get("zscore_period", 20)
        
        # Calculate rolling mean and standard deviation
        rolling_mean = spread.rolling(window=zscore_period).mean()
        rolling_std = spread.rolling(window=zscore_period).std()
        
        # Calculate z-score
        zscore = (spread - rolling_mean) / rolling_std
        self.zscores[pair_key] = zscore
    
    def _generate_pair_signals(self, pair_key: str, symbol: str, other_symbol: str) -> List[Signal]:
        """
        Generate signals for a pair based on z-score.
        
        Args:
            pair_key: Key for the pair
            symbol: Symbol to generate signal for
            other_symbol: Other symbol in the pair
            
        Returns:
            List of trading signals
        """
        signals = []
        
        if pair_key not in self.zscores:
            return signals
        
        # Get latest z-score
        zscore = self.zscores[pair_key].iloc[-1]
        if np.isnan(zscore):
            return signals
        
        # Get thresholds
        entry_threshold = self.parameters.get("entry_threshold", 2.0)
        exit_threshold = self.parameters.get("exit_threshold", 0.5)
        
        # Get current prices
        symbol_data = self.pair_data[symbol]
        other_data = self.pair_data[other_symbol]
        
        symbol_price = symbol_data['close'].iloc[-1]
        other_price = other_data['close'].iloc[-1]
        
        # Get hedge ratio
        hedge_ratio = self.hedge_ratios.get(pair_key, 1.0)
        
        # Get current positions
        symbol_position = self.current_positions.get(symbol, 0)
        other_position = self.current_positions.get(other_symbol, 0)
        
        # Check if we already have a position in this pair
        has_position = (symbol_position != 0) or (other_position != 0)
        
        # Generate signals
        if not has_position:
            # Entry signals
            if zscore > entry_threshold:
                # Spread is too high, short symbol, long other_symbol
                if symbol_position == 0:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        strength=SignalStrength.MODERATE,
                        price=symbol_price,
                        confidence=min(1.0, abs(zscore) / entry_threshold),
                        timestamp=datetime.now(),
                        metadata={
                            "strategy": "arbitrage",
                            "type": "statistical",
                            "pair": pair_key,
                            "other_symbol": other_symbol,
                            "zscore": zscore,
                            "hedge_ratio": hedge_ratio,
                            "action": "short_spread"
                        }
                    )
                    signals.append(signal)
                
                if other_position == 0:
                    signal = Signal(
                        symbol=other_symbol,
                        signal_type=SignalType.BUY,
                        strength=SignalStrength.MODERATE,
                        price=other_price,
                        confidence=min(1.0, abs(zscore) / entry_threshold),
                        timestamp=datetime.now(),
                        metadata={
                            "strategy": "arbitrage",
                            "type": "statistical",
                            "pair": pair_key,
                            "other_symbol": symbol,
                            "zscore": zscore,
                            "hedge_ratio": hedge_ratio,
                            "action": "short_spread"
                        }
                    )
                    signals.append(signal)
            
            elif zscore < -entry_threshold:
                # Spread is too low, long symbol, short other_symbol
                if symbol_position == 0:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        strength=SignalStrength.MODERATE,
                        price=symbol_price,
                        confidence=min(1.0, abs(zscore) / entry_threshold),
                        timestamp=datetime.now(),
                        metadata={
                            "strategy": "arbitrage",
                            "type": "statistical",
                            "pair": pair_key,
                            "other_symbol": other_symbol,
                            "zscore": zscore,
                            "hedge_ratio": hedge_ratio,
                            "action": "long_spread"
                        }
                    )
                    signals.append(signal)
                
                if other_position == 0:
                    signal = Signal(
                        symbol=other_symbol,
                        signal_type=SignalType.SELL,
                        strength=SignalStrength.MODERATE,
                        price=other_price,
                        confidence=min(1.0, abs(zscore) / entry_threshold),
                        timestamp=datetime.now(),
                        metadata={
                            "strategy": "arbitrage",
                            "type": "statistical",
                            "pair": pair_key,
                            "other_symbol": symbol,
                            "zscore": zscore,
                            "hedge_ratio": hedge_ratio,
                            "action": "long_spread"
                        }
                    )
                    signals.append(signal)
        
        else:
            # Exit signals
            if abs(zscore) < exit_threshold:
                # Spread has reverted to mean, close positions
                if symbol_position > 0:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        strength=SignalStrength.WEAK,
                        price=symbol_price,
                        confidence=0.7,
                        timestamp=datetime.now(),
                        metadata={
                            "strategy": "arbitrage",
                            "type": "statistical",
                            "pair": pair_key,
                            "other_symbol": other_symbol,
                            "zscore": zscore,
                            "action": "close_long"
                        }
                    )
                    signals.append(signal)
                
                elif symbol_position < 0:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        strength=SignalStrength.WEAK,
                        price=symbol_price,
                        confidence=0.7,
                        timestamp=datetime.now(),
                        metadata={
                            "strategy": "arbitrage",
                            "type": "statistical",
                            "pair": pair_key,
                            "other_symbol": other_symbol,
                            "zscore": zscore,
                            "action": "close_short"
                        }
                    )
                    signals.append(signal)
                
                if other_position > 0:
                    signal = Signal(
                        symbol=other_symbol,
                        signal_type=SignalType.SELL,
                        strength=SignalStrength.WEAK,
                        price=other_price,
                        confidence=0.7,
                        timestamp=datetime.now(),
                        metadata={
                            "strategy": "arbitrage",
                            "type": "statistical",
                            "pair": pair_key,
                            "other_symbol": symbol,
                            "zscore": zscore,
                            "action": "close_long"
                        }
                    )
                    signals.append(signal)
                
                elif other_position < 0:
                    signal = Signal(
                        symbol=other_symbol,
                        signal_type=SignalType.BUY,
                        strength=SignalStrength.WEAK,
                        price=other_price,
                        confidence=0.7,
                        timestamp=datetime.now(),
                        metadata={
                            "strategy": "arbitrage",
                            "type": "statistical",
                            "pair": pair_key,
                            "other_symbol": symbol,
                            "zscore": zscore,
                            "action": "close_short"
                        }
                    )
                    signals.append(signal)
        
        return signals
    
    def _generate_triangular_signals(self, tri_set: List[str], data: pd.DataFrame) -> List[Signal]:
        """
        Generate signals for triangular arbitrage.
        
        Args:
            tri_set: List of three symbols in the triangular set
            data: Aligned price data for the three symbols
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Get symbols
        symbol1, symbol2, symbol3 = tri_set
        
        # Get current prices
        price1 = data[symbol1].iloc[-1]
        price2 = data[symbol2].iloc[-1]
        price3 = data[symbol3].iloc[-1]
        
        # Calculate implied price
        # For a triangular arbitrage, we're looking for price1 * price2 * price3 ≈ 1
        # If it's greater than 1, we can profit by going around the triangle in one direction
        # If it's less than 1, we can profit by going around in the opposite direction
        
        implied_product = price1 * price2 * price3
        
        # Calculate potential profit after fees
        fee_rate = self.parameters.get("fee_rate", 0.001)
        min_profit_threshold = self.parameters.get("min_profit_threshold", 0.002)
        
        # Check for arbitrage opportunity
        if implied_product > 1 + min_profit_threshold + 3 * fee_rate:
            # Arbitrage opportunity in one direction
            profit_pct = implied_product - 1 - 3 * fee_rate
            
            # Generate signals for the three legs of the triangle
            # This is a simplified implementation
            # In a real implementation, we would need to determine the exact trading sequence
            
            for symbol in tri_set:
                current_price = data[symbol].iloc[-1]
                
                # Determine signal type based on position in the triangle
                # This is simplified - in reality, we'd need to determine the exact sequence
                signal_type = SignalType.BUY if np.random.random() > 0.5 else SignalType.SELL
                
                signal = Signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=SignalStrength.STRONG,
                    price=current_price,
                    confidence=min(1.0, profit_pct / min_profit_threshold),
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "arbitrage",
                        "type": "triangular",
                        "triangular_set": tri_set,
                        "implied_product": implied_product,
                        "profit_pct": profit_pct
                    }
                )
                signals.append(signal)
        
        elif implied_product < 1 - min_profit_threshold - 3 * fee_rate:
            # Arbitrage opportunity in the opposite direction
            profit_pct = 1 - implied_product - 3 * fee_rate
            
            # Generate signals for the three legs of the triangle
            for symbol in tri_set:
                current_price = data[symbol].iloc[-1]
                
                # Determine signal type based on position in the triangle
                # This is simplified - in reality, we'd need to determine the exact sequence
                signal_type = SignalType.BUY if np.random.random() > 0.5 else SignalType.SELL
                
                signal = Signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=SignalStrength.STRONG,
                    price=current_price,
                    confidence=min(1.0, profit_pct / min_profit_threshold),
                    timestamp=datetime.now(),
                    metadata={
                        "strategy": "arbitrage",
                        "type": "triangular",
                        "triangular_set": tri_set,
                        "implied_product": implied_product,
                        "profit_pct": profit_pct
                    }
                )
                signals.append(signal)
        
        return signals
    
    def on_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Called when a trade is executed.
        
        Args:
            trade_data: Trade execution data
        """
        # Call parent method
        super().on_trade(trade_data)
        
        # Update entry prices
        symbol = trade_data.get("symbol")
        if symbol:
            if symbol not in self.entry_prices:
                self.entry_prices[symbol] = {}
            
            self.entry_prices[symbol][datetime.now()] = trade_data.get("price", 0)