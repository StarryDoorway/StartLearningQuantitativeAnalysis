"""
Universal Technical Indicators module for the quantitative trading framework.

This module provides a comprehensive set of technical indicators that can be used
across different trading strategies, consolidating indicator calculations from
various strategy-specific modules.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Union, Optional


class TechnicalIndicators:
    """
    Universal Technical Indicators class.
    
    This class provides methods to calculate various technical indicators
    used in different trading strategies.
    """
    
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).
        
        Args:
            data: Price data
            period: SMA period
            
        Returns:
            SMA values
        """
        return data.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).
        
        Args:
            data: Price data
            period: EMA period
            
        Returns:
            EMA values
        """
        return data.ewm(span=period).mean()
    
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
    def calculate_bollinger_bands(data: pd.Series, period: int, dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: Price data
            period: Period for moving average and standard deviation
            dev: Number of standard deviations
            
        Returns:
            Tuple of (middle band, upper band, lower band)
        """
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std * dev)
        lower_band = sma - (std * dev)
        
        return sma, upper_band, lower_band
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast_period: int, slow_period: int, signal_period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            Tuple of (MACD line, signal line, histogram)
        """
        ema_fast = TechnicalIndicators.calculate_ema(data, fast_period)
        ema_slow = TechnicalIndicators.calculate_ema(data, slow_period)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high: High price data
            low: Low price data
            close: Close price data
            k_period: %K period
            d_period: %D period
            
        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            high: High price data
            low: Low price data
            close: Close price data
            period: ATR period
            
        Returns:
            ATR values
        """
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR using exponential moving average
        atr = true_range.ewm(span=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).
        
        Args:
            high: High price data
            low: Low price data
            close: Close price data
            period: ADX period
            
        Returns:
            ADX values
        """
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Convert to pandas Series
        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)
        
        # Calculate Smoothed True Range, Plus DM, and Minus DM
        atr = TechnicalIndicators.calculate_atr(high, low, close, period)
        plus_di = 100 * (plus_dm.ewm(span=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period).mean() / atr)
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period).mean()
        
        return adx
    
    @staticmethod
    def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Williams %R.
        
        Args:
            high: High price data
            low: Low price data
            close: Close price data
            period: Williams %R period
            
        Returns:
            Williams %R values
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    @staticmethod
    def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI).
        
        Args:
            high: High price data
            low: Low price data
            close: Close price data
            period: CCI period
            
        Returns:
            CCI values
        """
        # Calculate Typical Price
        typical_price = (high + low + close) / 3
        
        # Calculate Simple Moving Average of Typical Price
        sma_tp = typical_price.rolling(window=period).mean()
        
        # Calculate Mean Deviation
        mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        # Calculate CCI
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return cci
    
    @staticmethod
    def calculate_roc(data: pd.Series, period: int = 12) -> pd.Series:
        """
        Calculate Rate of Change (ROC).
        
        Args:
            data: Price data
            period: ROC period
            
        Returns:
            ROC values
        """
        roc = ((data - data.shift(period)) / data.shift(period)) * 100
        
        return roc
    
    @staticmethod
    def calculate_zscore(data: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Z-Score.
        
        Args:
            data: Price data
            period: Z-Score period
            
        Returns:
            Z-Score values
        """
        mean = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        zscore = (data - mean) / std
        
        return zscore
    
    @staticmethod
    def calculate_volatility(data: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate volatility (standard deviation of returns).
        
        Args:
            data: Price data
            period: Volatility period
            
        Returns:
            Volatility values
        """
        returns = data.pct_change()
        volatility = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
        
        return volatility
    
    @staticmethod
    def calculate_channel(high: pd.Series, low: pd.Series, period: int) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate price channel (highest high and lowest low).
        
        Args:
            high: High price data
            low: Low price data
            period: Channel period
            
        Returns:
            Tuple of (highest high, lowest low)
        """
        highest = high.rolling(window=period).max()
        lowest = low.rolling(window=period).min()
        
        return highest, lowest
    
    @staticmethod
    def calculate_cointegration(data1: pd.Series, data2: pd.Series) -> Tuple[float, float, pd.Series]:
        """
        Calculate cointegration between two price series.
        
        Args:
            data1: First price series
            data2: Second price series
            
        Returns:
            Tuple of (hedge ratio, p-value, spread)
        """
        try:
            import statsmodels.api as sm
            from statsmodels.tsa.stattools import coint
            
            # Align data
            aligned_data = pd.concat([data1, data2], axis=1, join='inner').dropna()
            if len(aligned_data) < 30:
                return 0.0, 1.0, pd.Series()
            
            x = aligned_data.iloc[:, 0]
            y = aligned_data.iloc[:, 1]
            
            # Calculate hedge ratio using linear regression
            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit()
            hedge_ratio = model.params[1]
            
            # Calculate spread
            spread = y - hedge_ratio * aligned_data.iloc[:, 0]
            
            # Test for cointegration
            coint_test = coint(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])
            p_value = coint_test[1]
            
            return hedge_ratio, p_value, spread
            
        except ImportError:
            # If statsmodels is not available, use simple correlation
            aligned_data = pd.concat([data1, data2], axis=1, join='inner').dropna()
            if len(aligned_data) < 30:
                return 0.0, 1.0, pd.Series()
                
            correlation = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
            hedge_ratio = correlation
            spread = aligned_data.iloc[:, 1] - hedge_ratio * aligned_data.iloc[:, 0]
            
            return hedge_ratio, 0.05, spread  # Assume p-value of 0.05 if correlation is significant
    
    @staticmethod
    def calculate_half_life(spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion for a spread.
        
        Args:
            spread: Spread series
            
        Returns:
            Half-life in periods
        """
        try:
            import statsmodels.api as sm
            
            # Calculate lagged spread
            lagged_spread = spread.shift(1).dropna()
            current_spread = spread.dropna()
            
            # Align data
            aligned_data = pd.concat([current_spread, lagged_spread], axis=1, join='inner').dropna()
            if len(aligned_data) < 30:
                return 10.0  # Default half-life
            
            y = aligned_data.iloc[:, 0] - aligned_data.iloc[:, 1]
            x = aligned_data.iloc[:, 1]
            
            # Calculate regression
            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit()
            theta = -model.params[1]
            
            # Calculate half-life
            if theta > 0:
                half_life = np.log(2) / theta
            else:
                half_life = 10.0  # Default half-life
                
            return half_life
            
        except ImportError:
            return 10.0  # Default half-life if statsmodels is not available