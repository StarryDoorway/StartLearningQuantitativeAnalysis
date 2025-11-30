"""
RSI Calculator module for the quantitative trading framework.

This module provides RSI calculation functionality for various strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class RsiCalculator:
    """
    RSI Calculator class.
    
    This class provides methods to calculate RSI and related indicators.
    """
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI indicator using the standard method.
        
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
        
        # Use exponential weighted moving average directly
        # This is the standard Wilder's smoothing method
        avg_gain = gain.ewm(com=period-1, adjust=False).mean()
        avg_loss = loss.ewm(com=period-1, adjust=False).mean()
        
        # Calculate RSI with proper handling of zero values
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Handle edge cases where avg_loss is 0 (all gains)
        rsi = rsi.where(avg_loss > 0, 100)
        
        # Handle edge cases where avg_gain is 0 (all losses)
        rsi = rsi.where(avg_gain > 0, 0)
        
        return rsi
    
    @staticmethod
    def calculate_rsi_with_signals(prices: pd.Series, period: int = 14, 
                                 overbought: float = 70, 
                                 oversold: float = 30) -> Dict[str, pd.Series]:
        """
        Calculate RSI with overbought/oversold signals.
        
        Args:
            prices: Price series
            period: RSI period
            overbought: Overbought threshold
            oversold: Oversold threshold
            
        Returns:
            Dictionary with RSI and signals
        """
        rsi = RsiCalculator.calculate_rsi(prices, period)
        
        # Generate signals
        overbought_signal = (rsi > overbought).astype(int)
        oversold_signal = (rsi < oversold).astype(int)
        
        return {
            'rsi': rsi,
            'overbought': overbought_signal,
            'oversold': oversold_signal
        }