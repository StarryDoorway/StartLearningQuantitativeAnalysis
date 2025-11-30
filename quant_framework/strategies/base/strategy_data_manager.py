"""
Strategy data management for the quantitative trading framework.

This module provides functionality for managing strategy data, including
market data updates, signal history, and trade history.
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime

from ...core.data_engine.data_engine import MarketData


logger = logging.getLogger(__name__)


class StrategyDataManager:
    """
    Manages data for a trading strategy.
    
    This class handles market data updates, signal history, and trade history.
    """
    
    def __init__(self, strategy_id: str, max_data_rows: int = 10000):
        """
        Initialize the data manager.
        
        Args:
            strategy_id: Strategy identifier
            max_data_rows: Maximum number of data rows to keep per symbol
        """
        self.strategy_id = strategy_id
        self.max_data_rows = max_data_rows
        
        # Data storage
        self.data = {}
        self.signal_history = []
        self.trade_history = []
        
        # Logger
        self.logger = logging.getLogger(f"{__name__}.{strategy_id}")
    
    def update_data(self, symbol: str, market_data: MarketData) -> None:
        """
        Update historical data with new market data.
        
        Args:
            symbol: Trading symbol
            market_data: New market data
        """
        # Convert market data to DataFrame row
        new_row = pd.DataFrame({
            "open": [market_data.open],
            "high": [market_data.high],
            "low": [market_data.low],
            "close": [market_data.close],
            "volume": [market_data.volume]
        }, index=[market_data.timestamp])
        
        # Add to historical data
        if symbol not in self.data:
            self.data[symbol] = new_row
        else:
            self.data[symbol] = pd.concat([self.data[symbol], new_row])
            
            # Keep only the last N rows to prevent memory issues
            if len(self.data[symbol]) > self.max_data_rows:
                self.data[symbol] = self.data[symbol].iloc[-self.max_data_rows:]
    
    def get_data(self, symbol: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get historical data for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Maximum number of rows to return
            
        Returns:
            Historical data
        """
        if symbol not in self.data:
            return pd.DataFrame()
        
        if limit:
            return self.data[symbol].iloc[-limit:]
        return self.data[symbol]
    
    def get_symbols(self) -> List[str]:
        """
        Get list of symbols the strategy is tracking.
        
        Returns:
            List of symbols
        """
        return list(self.data.keys())
    
    def add_signal(self, signal) -> None:
        """
        Add a signal to the signal history.
        
        Args:
            signal: Trading signal
        """
        self.signal_history.append(signal)
        
        # Keep only the last 1000 signals
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
        
        self.logger.debug(f"Added signal: {signal.signal_type.value} {signal.symbol}")
    
    def get_signal_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get signal history.
        
        Args:
            limit: Maximum number of signals to return
            
        Returns:
            Signal history
        """
        if limit:
            return [signal.to_dict() for signal in self.signal_history[-limit:]]
        return [signal.to_dict() for signal in self.signal_history]
    
    def add_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Add a trade to the trade history.
        
        Args:
            trade_data: Trade execution data
        """
        self.trade_history.append(trade_data)
        
        # Keep only the last 1000 trades
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
        
        self.logger.debug(f"Added trade: {trade_data}")
    
    def get_trade_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get trade history.
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            Trade history
        """
        if limit:
            return self.trade_history[-limit:]
        return self.trade_history
    
    def reset(self) -> None:
        """Reset all data."""
        self.data = {}
        self.signal_history = []
        self.trade_history = []
        self.logger.info(f"Reset all data for strategy {self.strategy_id}")