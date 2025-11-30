"""
Data processing utilities.

This module provides various utilities for processing market data, including
technical indicators, data cleaning, and data transformation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import math

from .data_sources import MarketData, DataFrequency

logger = logging.getLogger(__name__)


class IndicatorType(Enum):
    """Enumeration of technical indicator types."""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    OSCILLATOR = "oscillator"


class DataProcessor:
    """Data processor with various technical indicators and data utilities."""
    
    @staticmethod
    def clean_data(data: pd.DataFrame, 
                   remove_outliers: bool = True,
                   outlier_threshold: float = 3.0,
                   fill_missing: bool = True,
                   fill_method: str = "forward") -> pd.DataFrame:
        """
        Clean market data by handling outliers and missing values.
        
        Args:
            data: DataFrame with market data
            remove_outliers: Whether to remove outliers
            outlier_threshold: Z-score threshold for outlier detection
            fill_missing: Whether to fill missing values
            fill_method: Method to fill missing values ("forward", "backward", "interpolate")
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying the original
        cleaned_data = data.copy()
        
        # Ensure timestamp is the index
        if "timestamp" in cleaned_data.columns:
            cleaned_data = cleaned_data.set_index("timestamp")
        
        # Remove outliers
        if remove_outliers:
            price_columns = ["open", "high", "low", "close"]
            for col in price_columns:
                if col in cleaned_data.columns:
                    # Calculate Z-scores
                    z_scores = np.abs((cleaned_data[col] - cleaned_data[col].mean()) / cleaned_data[col].std())
                    
                    # Mark outliers as NaN
                    cleaned_data.loc[z_scores > outlier_threshold, col] = np.nan
        
        # Fill missing values
        if fill_missing:
            if fill_method == "forward":
                cleaned_data = cleaned_data.fillna(method="ffill")
            elif fill_method == "backward":
                cleaned_data = cleaned_data.fillna(method="bfill")
            elif fill_method == "interpolate":
                cleaned_data = cleaned_data.interpolate()
            
            # Fill any remaining NaN values with the previous value
            cleaned_data = cleaned_data.fillna(method="ffill")
        
        # Reset index to make timestamp a column again
        cleaned_data = cleaned_data.reset_index()
        
        return cleaned_data
    
    @staticmethod
    def resample_data(data: pd.DataFrame, target_frequency: DataFrequency) -> pd.DataFrame:
        """
        Resample data to a different frequency.
        
        Args:
            data: DataFrame with market data
            target_frequency: Target frequency
            
        Returns:
            Resampled DataFrame
        """
        # Make sure timestamp is the index
        if "timestamp" in data.columns and not isinstance(data.index, pd.DatetimeIndex):
            data = data.set_index("timestamp")
        
        # Map frequency to pandas resample string
        freq_map = {
            DataFrequency.SECOND: "1S",
            DataFrequency.MINUTE: "1T",
            DataFrequency.FIVE_MINUTE: "5T",
            DataFrequency.FIFTEEN_MINUTE: "15T",
            DataFrequency.THIRTY_MINUTE: "30T",
            DataFrequency.HOUR: "1H",
            DataFrequency.FOUR_HOUR: "4H",
            DataFrequency.DAY: "1D",
            DataFrequency.WEEK: "1W",
            DataFrequency.MONTH: "1M"
        }
        
        resample_freq = freq_map.get(target_frequency, "1T")
        
        # Resample OHLCV data
        resampled = data.resample(resample_freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })
        
        # Drop rows with NaN values
        resampled = resampled.dropna()
        
        # Reset index to make timestamp a column again
        resampled = resampled.reset_index()
        
        return resampled
    
    @staticmethod
    def calculate_returns(data: pd.DataFrame, 
                         return_type: str = "simple",
                         price_column: str = "close") -> pd.DataFrame:
        """
        Calculate returns for the given data.
        
        Args:
            data: DataFrame with market data
            return_type: Type of returns ("simple", "log")
            price_column: Column to use for price
            
        Returns:
            DataFrame with returns
        """
        # Make a copy to avoid modifying the original
        returns_data = data.copy()
        
        # Ensure timestamp is the index
        if "timestamp" in returns_data.columns:
            returns_data = returns_data.set_index("timestamp")
        
        if return_type == "simple":
            returns_data["return"] = returns_data[price_column].pct_change()
        elif return_type == "log":
            returns_data["return"] = np.log(returns_data[price_column] / returns_data[price_column].shift(1))
        else:
            raise ValueError(f"Unknown return type: {return_type}")
        
        # Reset index to make timestamp a column again
        returns_data = returns_data.reset_index()
        
        return returns_data
    
    @staticmethod
    def calculate_sma(data: pd.DataFrame, 
                      window: int = 20,
                      price_column: str = "close") -> pd.DataFrame:
        """
        Calculate Simple Moving Average (SMA).
        
        Args:
            data: DataFrame with market data
            window: Window size for SMA
            price_column: Column to use for price
            
        Returns:
            DataFrame with SMA
        """
        # Make a copy to avoid modifying the original
        sma_data = data.copy()
        
        # Calculate SMA
        sma_data[f"sma_{window}"] = sma_data[price_column].rolling(window=window).mean()
        
        return sma_data
    
    @staticmethod
    def calculate_ema(data: pd.DataFrame, 
                      window: int = 20,
                      price_column: str = "close") -> pd.DataFrame:
        """
        Calculate Exponential Moving Average (EMA).
        
        Args:
            data: DataFrame with market data
            window: Window size for EMA
            price_column: Column to use for price
            
        Returns:
            DataFrame with EMA
        """
        # Make a copy to avoid modifying the original
        ema_data = data.copy()
        
        # Calculate EMA
        ema_data[f"ema_{window}"] = ema_data[price_column].ewm(span=window).mean()
        
        return ema_data
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, 
                                 window: int = 20,
                                 num_std: float = 2.0,
                                 price_column: str = "close") -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: DataFrame with market data
            window: Window size for moving average
            num_std: Number of standard deviations
            price_column: Column to use for price
            
        Returns:
            DataFrame with Bollinger Bands
        """
        # Make a copy to avoid modifying the original
        bb_data = data.copy()
        
        # Calculate SMA
        bb_data[f"sma_{window}"] = bb_data[price_column].rolling(window=window).mean()
        
        # Calculate standard deviation
        bb_data[f"std_{window}"] = bb_data[price_column].rolling(window=window).std()
        
        # Calculate upper and lower bands
        bb_data[f"bb_upper_{window}"] = bb_data[f"sma_{window}"] + (bb_data[f"std_{window}"] * num_std)
        bb_data[f"bb_lower_{window}"] = bb_data[f"sma_{window}"] - (bb_data[f"std_{window}"] * num_std)
        
        # Calculate bandwidth
        bb_data[f"bb_bandwidth_{window}"] = (bb_data[f"bb_upper_{window}"] - bb_data[f"bb_lower_{window}"]) / bb_data[f"sma_{window}"]
        
        # Calculate %B
        bb_data[f"bb_percent_b_{window}"] = (bb_data[price_column] - bb_data[f"bb_lower_{window}"]) / (bb_data[f"bb_upper_{window}"] - bb_data[f"bb_lower_{window}"])
        
        return bb_data
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, 
                      window: int = 14,
                      price_column: str = "close") -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: DataFrame with market data
            window: Window size for RSI
            price_column: Column to use for price
            
        Returns:
            DataFrame with RSI
        """
        # Make a copy to avoid modifying the original
        rsi_data = data.copy()
        
        # Calculate price changes
        delta = rsi_data[price_column].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi_data[f"rsi_{window}"] = 100 - (100 / (1 + rs))
        
        return rsi_data
    
    @staticmethod
    def calculate_macd(data: pd.DataFrame, 
                       fast_period: int = 12,
                       slow_period: int = 26,
                       signal_period: int = 9,
                       price_column: str = "close") -> pd.DataFrame:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            data: DataFrame with market data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            price_column: Column to use for price
            
        Returns:
            DataFrame with MACD
        """
        # Make a copy to avoid modifying the original
        macd_data = data.copy()
        
        # Calculate EMAs
        ema_fast = macd_data[price_column].ewm(span=fast_period).mean()
        ema_slow = macd_data[price_column].ewm(span=slow_period).mean()
        
        # Calculate MACD line
        macd_data["macd"] = ema_fast - ema_slow
        
        # Calculate signal line
        macd_data["macd_signal"] = macd_data["macd"].ewm(span=signal_period).mean()
        
        # Calculate histogram
        macd_data["macd_histogram"] = macd_data["macd"] - macd_data["macd_signal"]
        
        return macd_data
    
    @staticmethod
    def calculate_stochastic(data: pd.DataFrame, 
                            k_window: int = 14,
                            d_window: int = 3) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            data: DataFrame with market data
            k_window: Window size for %K
            d_window: Window size for %D
            
        Returns:
            DataFrame with Stochastic Oscillator
        """
        # Make a copy to avoid modifying the original
        stoch_data = data.copy()
        
        # Calculate %K
        lowest_low = stoch_data["low"].rolling(window=k_window).min()
        highest_high = stoch_data["high"].rolling(window=k_window).max()
        stoch_data["stoch_k"] = 100 * ((stoch_data["close"] - lowest_low) / (highest_high - lowest_low))
        
        # Calculate %D
        stoch_data["stoch_d"] = stoch_data["stoch_k"].rolling(window=d_window).mean()
        
        return stoch_data
    
    @staticmethod
    def calculate_atr(data: pd.DataFrame, 
                      window: int = 14) -> pd.DataFrame:
        """
        Calculate Average True Range (ATR).
        
        Args:
            data: DataFrame with market data
            window: Window size for ATR
            
        Returns:
            DataFrame with ATR
        """
        # Make a copy to avoid modifying the original
        atr_data = data.copy()
        
        # Calculate True Range
        high_low = atr_data["high"] - atr_data["low"]
        high_close_prev = np.abs(atr_data["high"] - atr_data["close"].shift())
        low_close_prev = np.abs(atr_data["low"] - atr_data["close"].shift())
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate ATR
        atr_data[f"atr_{window}"] = true_range.rolling(window=window).mean()
        
        return atr_data
    
    @staticmethod
    def calculate_adx(data: pd.DataFrame, 
                      window: int = 14) -> pd.DataFrame:
        """
        Calculate Average Directional Index (ADX).
        
        Args:
            data: DataFrame with market data
            window: Window size for ADX
            
        Returns:
            DataFrame with ADX
        """
        # Make a copy to avoid modifying the original
        adx_data = data.copy()
        
        # Calculate True Range
        high_low = adx_data["high"] - adx_data["low"]
        high_close_prev = np.abs(adx_data["high"] - adx_data["close"].shift())
        low_close_prev = np.abs(adx_data["low"] - adx_data["close"].shift())
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate directional movements
        up_move = adx_data["high"] - adx_data["high"].shift()
        down_move = adx_data["low"].shift() - adx_data["low"]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Calculate smoothed values
        tr_smooth = true_range.rolling(window=window).sum()
        plus_dm_smooth = pd.Series(plus_dm).rolling(window=window).sum()
        minus_dm_smooth = pd.Series(minus_dm).rolling(window=window).sum()
        
        # Calculate directional indicators
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        # Calculate ADX
        dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
        adx_data[f"adx_{window}"] = dx.rolling(window=window).mean()
        adx_data[f"plus_di_{window}"] = plus_di
        adx_data[f"minus_di_{window}"] = minus_di
        
        return adx_data
    
    @staticmethod
    def calculate_vwap(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with VWAP
        """
        # Make a copy to avoid modifying the original
        vwap_data = data.copy()
        
        # Calculate typical price
        vwap_data["typical_price"] = (vwap_data["high"] + vwap_data["low"] + vwap_data["close"]) / 3
        
        # Calculate cumulative values
        vwap_data["cumulative_tp_volume"] = (vwap_data["typical_price"] * vwap_data["volume"]).cumsum()
        vwap_data["cumulative_volume"] = vwap_data["volume"].cumsum()
        
        # Calculate VWAP
        vwap_data["vwap"] = vwap_data["cumulative_tp_volume"] / vwap_data["cumulative_volume"]
        
        return vwap_data
    
    @staticmethod
    def calculate_returns_distribution(data: pd.DataFrame,
                                     price_column: str = "close",
                                     bins: int = 50) -> Dict[str, Any]:
        """
        Calculate returns distribution statistics.
        
        Args:
            data: DataFrame with market data
            price_column: Column to use for price
            bins: Number of bins for histogram
            
        Returns:
            Dictionary with distribution statistics
        """
        # Calculate returns
        returns = data[price_column].pct_change().dropna()
        
        # Calculate statistics
        stats = {
            "mean": returns.mean(),
            "std": returns.std(),
            "skew": returns.skew(),
            "kurtosis": returns.kurtosis(),
            "min": returns.min(),
            "max": returns.max(),
            "median": returns.median(),
            "q25": returns.quantile(0.25),
            "q75": returns.quantile(0.75),
            "histogram": np.histogram(returns, bins=bins),
            "sharpe_ratio": returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        }
        
        return stats
    
    @staticmethod
    def calculate_correlation_matrix(data: pd.DataFrame,
                                    symbols: List[str],
                                    price_column: str = "close") -> pd.DataFrame:
        """
        Calculate correlation matrix for multiple symbols.
        
        Args:
            data: DataFrame with market data for multiple symbols
            symbols: List of symbols to include in correlation matrix
            price_column: Column to use for price
            
        Returns:
            Correlation matrix DataFrame
        """
        # Create a DataFrame with returns for each symbol
        returns_df = pd.DataFrame()
        
        for symbol in symbols:
            symbol_data = data[data["symbol"] == symbol].copy()
            if not symbol_data.empty:
                symbol_data = symbol_data.sort_values("timestamp")
                symbol_returns = symbol_data[price_column].pct_change().dropna()
                returns_df[symbol] = symbol_returns
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix
    
    @staticmethod
    def detect_regimes(data: pd.DataFrame,
                      window: int = 252,
                      price_column: str = "close") -> pd.DataFrame:
        """
        Detect market regimes based on volatility and trend.
        
        Args:
            data: DataFrame with market data
            window: Window size for regime detection
            price_column: Column to use for price
            
        Returns:
            DataFrame with regime labels
        """
        # Make a copy to avoid modifying the original
        regime_data = data.copy()
        
        # Calculate returns
        regime_data["return"] = regime_data[price_column].pct_change()
        
        # Calculate rolling volatility
        regime_data["volatility"] = regime_data["return"].rolling(window=window).std()
        
        # Calculate rolling trend (slope of price over window)
        regime_data["trend"] = regime_data[price_column].rolling(window=window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
        
        # Classify regimes
        # High volatility: volatility > 75th percentile
        # Low volatility: volatility < 25th percentile
        # Bull market: trend > 0
        # Bear market: trend < 0
        
        vol_high = regime_data["volatility"].quantile(0.75)
        vol_low = regime_data["volatility"].quantile(0.25)
        
        conditions = [
            (regime_data["volatility"] > vol_high) & (regime_data["trend"] > 0),  # High volatility, bull
            (regime_data["volatility"] > vol_high) & (regime_data["trend"] <= 0),  # High volatility, bear
            (regime_data["volatility"] <= vol_high) & (regime_data["volatility"] >= vol_low) & (regime_data["trend"] > 0),  # Medium volatility, bull
            (regime_data["volatility"] <= vol_high) & (regime_data["volatility"] >= vol_low) & (regime_data["trend"] <= 0),  # Medium volatility, bear
            (regime_data["volatility"] < vol_low) & (regime_data["trend"] > 0),  # Low volatility, bull
            (regime_data["volatility"] < vol_low) & (regime_data["trend"] <= 0),  # Low volatility, bear
        ]
        
        choices = [
            "High Volatility Bull",
            "High Volatility Bear",
            "Medium Volatility Bull",
            "Medium Volatility Bear",
            "Low Volatility Bull",
            "Low Volatility Bear"
        ]
        
        regime_data["regime"] = np.select(conditions, choices, default="Unknown")
        
        return regime_data