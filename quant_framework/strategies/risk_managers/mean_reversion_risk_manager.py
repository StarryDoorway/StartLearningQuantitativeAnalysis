"""
Risk manager for mean reversion strategy.

This module contains risk management methods used by the mean reversion strategy.
"""

import logging
from typing import Dict, List, Any

from ..base.signal_types import Signal, SignalType

logger = logging.getLogger(__name__)


class MeanReversionRiskManager:
    """
    Risk manager for mean reversion strategy.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize the risk manager.
        
        Args:
            parameters: Strategy parameters
        """
        self.parameters = parameters
        self.logger = logger
        self.entry_prices: Dict[str, float] = {}
        self.stop_losses: Dict[str, float] = {}
        self.take_profits: Dict[str, float] = {}
    
    def apply_position_limits(self, signals: List[Signal], current_positions: Dict[str, float], indicators: Dict[str, Dict[str, Any]]) -> List[Signal]:
        """
        Apply position limits to signals.
        
        Args:
            signals: List of signals
            current_positions: Current positions
            indicators: Technical indicators
            
        Returns:
            Filtered list of signals
        """
        # Check maximum number of positions
        max_positions = self.parameters.get("max_positions", 5)
        current_position_count = len([pos for pos in current_positions.values() if pos != 0])
        
        if current_position_count >= max_positions:
            # Only allow exit signals
            return [signal for signal in signals if 
                   (signal.signal_type == SignalType.BUY and current_positions.get(signal.symbol, 0) < 0) or
                   (signal.signal_type == SignalType.SELL and current_positions.get(signal.symbol, 0) > 0)]
        
        # Apply volatility adjustment to position size
        if self.parameters.get("volatility_adjustment", True):
            for signal in signals:
                symbol = signal.symbol
                if symbol in indicators and 'volatility' in indicators[symbol]:
                    volatility = indicators[symbol]['volatility'].iloc[-1]
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
    
    def get_entry_price(self, symbol: str) -> float:
        """
        Get entry price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Entry price
        """
        return self.entry_prices.get(symbol, 0.0)
    
    def get_stop_loss(self, symbol: str) -> float:
        """
        Get stop loss price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Stop loss price
        """
        return self.stop_losses.get(symbol, 0.0)
    
    def get_take_profit(self, symbol: str) -> float:
        """
        Get take profit price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Take profit price
        """
        return self.take_profits.get(symbol, 0.0)