"""
Data source interfaces and implementations.

This module provides interfaces and implementations for various data sources
used in the quantitative trading framework.
"""

import logging
import abc
import asyncio
import time
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from enum import Enum

from ..utils.config_loader import get_config

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Enumeration of supported data source types."""
    EXCHANGE_API = "exchange_api"
    CSV_FILE = "csv_file"
    DATABASE = "database"
    WEBSOCKET = "websocket"
    REST_API = "rest_api"


class DataFrequency(Enum):
    """Enumeration of supported data frequencies."""
    TICK = "tick"
    SECOND = "1s"
    MINUTE = "1m"
    FIVE_MINUTE = "5m"
    FIFTEEN_MINUTE = "15m"
    THIRTY_MINUTE = "30m"
    HOUR = "1h"
    FOUR_HOUR = "4h"
    DAY = "1d"
    WEEK = "1w"
    MONTH = "1M"


class MarketData:
    """Container for market data."""
    
    def __init__(
        self,
        symbol: str,
        timestamp: datetime,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float,
        frequency: DataFrequency = DataFrequency.MINUTE,
        exchange: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize market data.
        
        Args:
            symbol: Trading symbol
            timestamp: Data timestamp
            open_price: Opening price
            high_price: Highest price
            low_price: Lowest price
            close_price: Closing price
            volume: Trading volume
            frequency: Data frequency
            exchange: Exchange name
            additional_data: Additional data fields
        """
        self.symbol = symbol
        self.timestamp = timestamp
        self.open_price = open_price
        self.high_price = high_price
        self.low_price = low_price
        self.close_price = close_price
        self.volume = volume
        self.frequency = frequency
        self.exchange = exchange
        self.additional_data = additional_data or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert market data to dictionary."""
        data = {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "open": self.open_price,
            "high": self.high_price,
            "low": self.low_price,
            "close": self.close_price,
            "volume": self.volume,
            "frequency": self.frequency.value,
            "exchange": self.exchange
        }
        data.update(self.additional_data)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketData':
        """Create market data from dictionary."""
        additional_data = {k: v for k, v in data.items() 
                          if k not in ["symbol", "timestamp", "open", "high", "low", "close", "volume", "frequency", "exchange"]}
        
        return cls(
            symbol=data["symbol"],
            timestamp=data["timestamp"],
            open_price=data["open"],
            high_price=data["high"],
            low_price=data["low"],
            close_price=data["close"],
            volume=data["volume"],
            frequency=DataFrequency(data.get("frequency", "1m")),
            exchange=data.get("exchange"),
            additional_data=additional_data
        )
    
    def to_dataframe_row(self) -> Dict[str, Any]:
        """Convert market data to a DataFrame row."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "open": self.open_price,
            "high": self.high_price,
            "low": self.low_price,
            "close": self.close_price,
            "volume": self.volume,
            "exchange": self.exchange
        }


class DataSource(abc.ABC):
    """Abstract base class for data sources."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data source.
        
        Args:
            config: Data source configuration
        """
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
        self.type = DataSourceType(config.get("type", "rest_api"))
        self.is_connected = False
        self.last_update = None
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
    
    @abc.abstractmethod
    def connect(self) -> bool:
        """
        Connect to the data source.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the data source."""
        pass
    
    @abc.abstractmethod
    def is_data_available(self, symbol: str, frequency: DataFrequency, 
                         start_time: datetime, end_time: datetime) -> bool:
        """
        Check if data is available for the given parameters.
        
        Args:
            symbol: Trading symbol
            frequency: Data frequency
            start_time: Start time
            end_time: End time
            
        Returns:
            True if data is available, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        frequency: DataFrequency,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get historical market data.
        
        Args:
            symbol: Trading symbol
            frequency: Data frequency
            start_time: Start time
            end_time: End time
            limit: Maximum number of data points
            
        Returns:
            DataFrame with historical data
        """
        pass
    
    @abc.abstractmethod
    def subscribe_to_realtime_data(
        self,
        symbols: List[str],
        frequency: DataFrequency,
        callback: Callable[[MarketData], None]
    ) -> bool:
        """
        Subscribe to real-time market data.
        
        Args:
            symbols: List of symbols to subscribe to
            frequency: Data frequency
            callback: Callback function for data updates
            
        Returns:
            True if subscription successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def unsubscribe_from_realtime_data(self, symbols: List[str]) -> bool:
        """
        Unsubscribe from real-time market data.
        
        Args:
            symbols: List of symbols to unsubscribe from
            
        Returns:
            True if unsubscription successful, False otherwise
        """
        pass
    
    def get_latest_data(self, symbol: str, frequency: DataFrequency, 
                       count: int = 1) -> pd.DataFrame:
        """
        Get the latest market data.
        
        Args:
            symbol: Trading symbol
            frequency: Data frequency
            count: Number of latest data points
            
        Returns:
            DataFrame with latest data
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)  # Default to 7 days ago
        
        # Get historical data and return the latest count rows
        df = self.get_historical_data(symbol, frequency, start_time, end_time)
        return df.tail(count)
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate market data.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            True if data is valid, False otherwise
        """
        # Check if DataFrame is empty
        if data.empty:
            return False
        
        # Check required columns
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in data.columns:
                self.logger.error(f"Missing required column: {col}")
                return False
        
        # Check for NaN values
        if data[required_columns].isnull().any().any():
            self.logger.warning("Data contains NaN values")
        
        # Check for negative prices or volume
        if (data[["open", "high", "low", "close", "volume"]] < 0).any().any():
            self.logger.warning("Data contains negative prices or volume")
        
        # Check for price consistency (high >= low, high >= open, high >= close, low <= open, low <= close)
        price_cols = ["open", "high", "low", "close"]
        if not (data["high"] >= data["low"]).all():
            self.logger.warning("High price is less than low price")
        
        if not (data["high"] >= data[["open", "close"]].max(axis=1)).all():
            self.logger.warning("High price is less than open or close price")
        
        if not (data["low"] <= data[["open", "close"]].min(axis=1)).all():
            self.logger.warning("Low price is greater than open or close price")
        
        return True
    
    def resample_data(self, data: pd.DataFrame, target_frequency: DataFrequency) -> pd.DataFrame:
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
    
    def get_supported_symbols(self) -> List[str]:
        """
        Get list of supported symbols.
        
        Returns:
            List of supported symbols
        """
        return self.config.get("supported_symbols", [])
    
    def get_supported_frequencies(self) -> List[DataFrequency]:
        """
        Get list of supported frequencies.
        
        Returns:
            List of supported frequencies
        """
        supported_freqs = self.config.get("supported_frequencies", ["1m", "5m", "15m", "1h", "1d"])
        return [DataFrequency(freq) for freq in supported_freqs]


class CSVDataSource(DataSource):
    """CSV file data source implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the CSV data source.
        
        Args:
            config: Data source configuration
        """
        config["type"] = "csv_file"
        super().__init__(config)
        self.data_cache = {}
        self.file_paths = config.get("file_paths", {})
        self.data_dir = config.get("data_dir", "")
    
    def connect(self) -> bool:
        """
        Connect to the data source.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Check if data directory exists
            if self.data_dir and not os.path.exists(self.data_dir):
                self.logger.error(f"Data directory does not exist: {self.data_dir}")
                return False
            
            self.is_connected = True
            self.last_update = datetime.now()
            self.logger.info("Connected to CSV data source")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to CSV data source: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the data source."""
        self.is_connected = False
        self.data_cache.clear()
        self.logger.info("Disconnected from CSV data source")
    
    def is_data_available(self, symbol: str, frequency: DataFrequency, 
                         start_time: datetime, end_time: datetime) -> bool:
        """
        Check if data is available for the given parameters.
        
        Args:
            symbol: Trading symbol
            frequency: Data frequency
            start_time: Start time
            end_time: End time
            
        Returns:
            True if data is available, False otherwise
        """
        # Check if we have a file path for this symbol
        file_path = self._get_file_path(symbol, frequency)
        if not file_path or not os.path.exists(file_path):
            return False
        
        try:
            # Load a small sample of data to check date range
            sample_data = pd.read_csv(file_path, nrows=10)
            if "timestamp" not in sample_data.columns:
                return False
            
            # Convert timestamp column to datetime
            sample_data["timestamp"] = pd.to_datetime(sample_data["timestamp"])
            
            # Check if the date range overlaps with requested range
            data_start = sample_data["timestamp"].min()
            data_end = sample_data["timestamp"].max()
            
            return not (data_end < start_time or data_start > end_time)
        except Exception as e:
            self.logger.error(f"Error checking data availability: {e}")
            return False
    
    def get_historical_data(
        self,
        symbol: str,
        frequency: DataFrequency,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get historical market data.
        
        Args:
            symbol: Trading symbol
            frequency: Data frequency
            start_time: Start time
            end_time: End time
            limit: Maximum number of data points
            
        Returns:
            DataFrame with historical data
        """
        # Check if data is cached
        cache_key = f"{symbol}_{frequency.value}"
        if cache_key in self.data_cache:
            data = self.data_cache[cache_key]
        else:
            # Load data from file
            file_path = self._get_file_path(symbol, frequency)
            if not file_path or not os.path.exists(file_path):
                self.logger.error(f"Data file not found for {symbol}")
                return pd.DataFrame()
            
            try:
                data = pd.read_csv(file_path)
                
                # Convert timestamp column to datetime
                if "timestamp" in data.columns:
                    data["timestamp"] = pd.to_datetime(data["timestamp"])
                
                # Validate data
                if not self.validate_data(data):
                    self.logger.error(f"Invalid data in file: {file_path}")
                    return pd.DataFrame()
                
                # Cache the data
                self.data_cache[cache_key] = data
            except Exception as e:
                self.logger.error(f"Error loading data from file: {e}")
                return pd.DataFrame()
        
        # Filter by date range
        if "timestamp" in data.columns:
            mask = (data["timestamp"] >= start_time) & (data["timestamp"] <= end_time)
            filtered_data = data.loc[mask]
        else:
            filtered_data = data
        
        # Apply limit if specified
        if limit and len(filtered_data) > limit:
            filtered_data = filtered_data.tail(limit)
        
        return filtered_data
    
    def subscribe_to_realtime_data(
        self,
        symbols: List[str],
        frequency: DataFrequency,
        callback: Callable[[MarketData], None]
    ) -> bool:
        """
        Subscribe to real-time market data.
        
        Args:
            symbols: List of symbols to subscribe to
            frequency: Data frequency
            callback: Callback function for data updates
            
        Returns:
            True if subscription successful, False otherwise
        """
        # CSV data source doesn't support real-time data
        self.logger.warning("CSV data source does not support real-time data")
        return False
    
    def unsubscribe_from_realtime_data(self, symbols: List[str]) -> bool:
        """
        Unsubscribe from real-time market data.
        
        Args:
            symbols: List of symbols to unsubscribe from
            
        Returns:
            True if unsubscription successful, False otherwise
        """
        # CSV data source doesn't support real-time data
        self.logger.warning("CSV data source does not support real-time data")
        return False
    
    def _get_file_path(self, symbol: str, frequency: DataFrequency) -> Optional[str]:
        """
        Get file path for a symbol and frequency.
        
        Args:
            symbol: Trading symbol
            frequency: Data frequency
            
        Returns:
            File path or None if not found
        """
        # Check if we have a specific file path for this symbol
        symbol_key = f"{symbol}_{frequency.value}"
        if symbol_key in self.file_paths:
            return self.file_paths[symbol_key]
        
        # Check if we have a file path for this symbol
        if symbol in self.file_paths:
            return self.file_paths[symbol]
        
        # Construct file path based on data directory and symbol
        if self.data_dir:
            filename = f"{symbol}_{frequency.value}.csv"
            return os.path.join(self.data_dir, filename)
        
        return None


# Import os for file operations
import os