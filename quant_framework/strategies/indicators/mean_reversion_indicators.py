"""
Technical indicators for mean reversion strategy.

This module contains technical indicators calculation methods used by the mean reversion strategy.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger(__name__)


class MeanReversionIndicators:
    """
    Technical indicators calculator for mean reversion strategy.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize the indicators calculator.
        
        Args:
            parameters: Strategy parameters
        """
        self.parameters = parameters
        self.indicators: Dict[str, Dict[str, pd.Series]] = {}
        self.logger = logger
    
    def calculate_indicators(self, symbol: str, data: pd.DataFrame) -> None:
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
            self._calculate_bollinger_bands(symbol, data)
        
        # Calculate RSI if enabled
        if self.parameters.get("use_rsi", False):
            self._calculate_rsi(symbol, data)
        
        # Calculate volatility for position sizing
        indicators['volatility'] = data['close'].pct_change().rolling(window=lookback).std()
    
    def _calculate_bollinger_bands(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Calculate Bollinger Bands indicators.
        
        Args:
            symbol: Trading symbol
            data: Historical market data
        """
        indicators = self.indicators[symbol]
        bb_period = self.parameters.get("bb_period", 20)
        bb_std = self.parameters.get("bb_std", 2.0)
        
        indicators['bb_middle'] = data['close'].rolling(window=bb_period).mean()
        indicators['bb_std'] = data['close'].rolling(window=bb_period).std()
        indicators['bb_upper'] = indicators['bb_middle'] + (indicators['bb_std'] * bb_std)
        indicators['bb_lower'] = indicators['bb_middle'] - (indicators['bb_std'] * bb_std)
        indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle']
        indicators['bb_position'] = (data['close'] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
    
    def _calculate_rsi(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Calculate RSI indicator.
        
        Args:
            symbol: Trading symbol
            data: Historical market data
        """
        indicators = self.indicators[symbol]
        rsi_period = self.parameters.get("rsi_period", 14)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs))
    
    def get_indicators(self, symbol: str) -> Dict[str, pd.Series]:
        """
        Get indicators for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of indicators
        """
        return self.indicators.get(symbol, {})
    
    def is_band_widening(self, symbol: str) -> bool:
        """
        Check if Bollinger Bands are widening.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if bands are widening
        """
        indicators = self.indicators.get(symbol, {})
        if 'bb_width' not in indicators or len(indicators['bb_width']) < 2:
            return False
        
        # Check if current width is greater than previous width
        current_width = indicators['bb_width'].iloc[-1]
        previous_width = indicators['bb_width'].iloc[-2]
        
        return current_width > previous_width