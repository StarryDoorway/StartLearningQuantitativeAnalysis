"""
RSI Divergence Indicators module for the quantitative trading framework.

This module contains the calculation logic for RSI and related indicators
used in RSI divergence trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple


class RsiDivergenceIndicators:
    """
    Class for calculating RSI and related indicators for divergence analysis.
    
    This class provides methods to calculate RSI and track price/RSI extremes
    for detecting divergences in trading strategies.
    """
    
    def __init__(self, rsi_period: int = 14):
        """
        Initialize the RSI Divergence Indicators calculator.
        
        Args:
            rsi_period: Period for RSI calculation
        """
        self.rsi_period = rsi_period
        
        # Initialize state variables for tracking extremes
        self.last_rsi_low = {}
        self.last_price_low = {}
        self.last_rsi_high = {}
        self.last_price_high = {}
    
    def calculate_rsi(self, prices: pd.Series, period: Optional[int] = None) -> pd.Series:
        """
        Calculate RSI indicator.
        
        Args:
            prices: Price series
            period: RSI period (uses default if None)
            
        Returns:
            RSI series
        """
        if period is None:
            period = self.rsi_period
            
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
    
    def update_extremes(self, symbol: str, data: pd.DataFrame, rsi: pd.Series) -> None:
        """
        Update RSI and price extremes for divergence detection.
        
        Args:
            symbol: Trading symbol
            data: Price data
            rsi: RSI values
        """
        current_rsi = rsi.iloc[-1]
        current_price = data["close"].iloc[-1]
        
        # Initialize symbol-specific state if needed
        if symbol not in self.last_rsi_low:
            self.last_rsi_low[symbol] = None
            self.last_price_low[symbol] = None
            self.last_rsi_high[symbol] = None
            self.last_price_high[symbol] = None
        
        # Update RSI and price lows
        if self.last_rsi_low[symbol] is None or current_rsi < self.last_rsi_low[symbol]:
            self.last_rsi_low[symbol] = current_rsi
            self.last_price_low[symbol] = current_price
        
        # Update RSI and price highs
        if self.last_rsi_high[symbol] is None or current_rsi > self.last_rsi_high[symbol]:
            self.last_rsi_high[symbol] = current_rsi
            self.last_price_high[symbol] = current_price
    
    def get_extremes(self, symbol: str) -> Dict[str, Optional[float]]:
        """
        Get the current extremes for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with current extremes
        """
        return {
            "last_rsi_low": self.last_rsi_low.get(symbol),
            "last_price_low": self.last_price_low.get(symbol),
            "last_rsi_high": self.last_rsi_high.get(symbol),
            "last_price_high": self.last_price_high.get(symbol)
        }
    
    def reset_extremes(self, symbol: str) -> None:
        """
        Reset the extremes for a symbol.
        
        Args:
            symbol: Trading symbol
        """
        if symbol in self.last_rsi_low:
            self.last_rsi_low[symbol] = None
            self.last_price_low[symbol] = None
            self.last_rsi_high[symbol] = None
            self.last_price_high[symbol] = None
    
    def check_bullish_divergence(self, symbol: str, data: pd.DataFrame, rsi: pd.Series,
                                rsi_oversold: float = 30, divergence_lookback: int = 5,
                                price_change_threshold: float = 0.005,
                                rsi_change_threshold: float = 2.0) -> bool:
        """
        Check for bullish divergence (buy signal).
        
        Args:
            symbol: Trading symbol
            data: Price data
            rsi: RSI values
            rsi_oversold: RSI oversold threshold
            divergence_lookback: Number of bars to look back for divergence
            price_change_threshold: Minimum price change threshold
            rsi_change_threshold: Minimum RSI change threshold
            
        Returns:
            True if bullish divergence is detected
        """
        # RSI must be in oversold territory
        if rsi.iloc[-1] >= rsi_oversold:
            return False
        
        # Check if we have enough data
        if len(data) < divergence_lookback + 1 or len(rsi) < divergence_lookback + 1:
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
        
        # Confirm bullish divergence
        return (rsi_decreasing and rsi_still_decreasing and 
                price_not_decreasing and rsi_significant_change)
    
    def check_bearish_divergence(self, symbol: str, data: pd.DataFrame, rsi: pd.Series,
                                rsi_overbought: float = 70, divergence_lookback: int = 5,
                                price_change_threshold: float = 0.005,
                                rsi_change_threshold: float = 2.0) -> bool:
        """
        Check for bearish divergence (sell signal).
        
        Args:
            symbol: Trading symbol
            data: Price data
            rsi: RSI values
            rsi_overbought: RSI overbought threshold
            divergence_lookback: Number of bars to look back for divergence
            price_change_threshold: Minimum price change threshold
            rsi_change_threshold: Minimum RSI change threshold
            
        Returns:
            True if bearish divergence is detected
        """
        # RSI must be in overbought territory
        if rsi.iloc[-1] <= rsi_overbought:
            return False
        
        # Check if we have enough data
        if len(data) < divergence_lookback + 1 or len(rsi) < divergence_lookback + 1:
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
        
        # Confirm bearish divergence
        return (rsi_increasing and rsi_still_increasing and 
                price_not_increasing and rsi_significant_change)
    
    def calculate_indicators(self, symbol: str, data: pd.DataFrame, 
                           params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate all necessary indicators for RSI divergence analysis.
        
        Args:
            symbol: Trading symbol
            data: Price data
            params: Strategy parameters
            
        Returns:
            Dictionary with calculated indicators
        """
        # Get parameters
        rsi_period = params.get("rsi_period", self.rsi_period)
        
        # Calculate RSI
        rsi = self.calculate_rsi(data["close"], rsi_period)
        
        # Update extremes for divergence detection
        self.update_extremes(symbol, data, rsi)
        
        # Get current values
        current_price = data["close"].iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        # Return indicators
        return {
            "rsi": rsi,
            "current_price": current_price,
            "current_rsi": current_rsi,
            "extremes": self.get_extremes(symbol)
        }