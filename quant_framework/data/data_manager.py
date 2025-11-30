"""
Data manager for handling data sources and data access.

This module provides a centralized data manager that handles multiple data sources,
caches data, and provides a unified interface for data access.
"""

import logging
import asyncio
import threading
import time
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from enum import Enum
import os
import json
from collections import defaultdict

from .data_sources import (
    DataSource, DataSourceType, DataFrequency, MarketData,
    CSVDataSource
)
from ..utils.config_loader import get_config
from ..core.event_bus import EventBus, EventType, Event, get_event_bus

logger = logging.getLogger(__name__)


class DataManagerStatus(Enum):
    """Enumeration of data manager status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class DataCache:
    """Data cache implementation."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize the data cache.
        
        Args:
            max_size: Maximum number of items in cache
            ttl_seconds: Time to live for cache items in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.timestamps = {}
        self.access_times = {}
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """
        Get data from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not found or expired
        """
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check if item is expired
            if time.time() - self.timestamps[key] > self.ttl_seconds:
                self._remove(key)
                return None
            
            # Update access time
            self.access_times[key] = time.time()
            return self.cache[key].copy()
    
    def put(self, key: str, data: pd.DataFrame) -> None:
        """
        Put data into cache.
        
        Args:
            key: Cache key
            data: Data to cache
        """
        with self.lock:
            # If cache is full, remove least recently used item
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._remove_lru()
            
            current_time = time.time()
            self.cache[key] = data.copy()
            self.timestamps[key] = current_time
            self.access_times[key] = current_time
    
    def clear(self) -> None:
        """Clear all items from cache."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.access_times.clear()
    
    def _remove(self, key: str) -> None:
        """
        Remove item from cache.
        
        Args:
            key: Cache key to remove
        """
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]
            del self.access_times[key]
    
    def _remove_lru(self) -> None:
        """Remove least recently used item from cache."""
        if not self.access_times:
            return
        
        # Find key with oldest access time
        lru_key = min(self.access_times, key=self.access_times.get)
        self._remove(lru_key)


class DataManager:
    """Centralized data manager for handling data sources and data access."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config().get("data", {})
        self.status = DataManagerStatus.STOPPED
        self.data_sources = {}
        self.data_cache = DataCache(
            max_size=self.config.get("cache_max_size", 1000),
            ttl_seconds=self.config.get("cache_ttl_seconds", 3600)
        )
        self.subscriptions = defaultdict(list)  # symbol -> list of callbacks
        self.event_bus = get_event_bus()
        self.logger = logging.getLogger(__name__)
        
        # Initialize data sources
        self._initialize_data_sources()
    
    def _initialize_data_sources(self) -> None:
        """Initialize data sources from configuration."""
        sources_config = self.config.get("sources", {})
        
        for source_name, source_config in sources_config.items():
            try:
                source_type = source_config.get("type", "csv_file")
                
                if source_type == "csv_file":
                    data_source = CSVDataSource(source_config)
                else:
                    self.logger.warning(f"Unsupported data source type: {source_type}")
                    continue
                
                self.data_sources[source_name] = data_source
                self.logger.info(f"Initialized data source: {source_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize data source {source_name}: {e}")
    
    def start(self) -> bool:
        """
        Start the data manager.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.status != DataManagerStatus.STOPPED:
            self.logger.warning(f"Data manager is not stopped (status: {self.status})")
            return False
        
        self.status = DataManagerStatus.STARTING
        
        try:
            # Connect to all data sources
            for source_name, data_source in self.data_sources.items():
                if not data_source.connect():
                    self.logger.error(f"Failed to connect to data source: {source_name}")
                    self.status = DataManagerStatus.ERROR
                    return False
            
            self.status = DataManagerStatus.RUNNING
            self.logger.info("Data manager started successfully")
            
            # Emit event
            self.event_bus.publish(Event(
                event_type=EventType.SYSTEM,
                data={"message": "Data manager started", "status": self.status.value}
            ))
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to start data manager: {e}")
            self.status = DataManagerStatus.ERROR
            return False
    
    def stop(self) -> bool:
        """
        Stop the data manager.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if self.status != DataManagerStatus.RUNNING:
            self.logger.warning(f"Data manager is not running (status: {self.status})")
            return False
        
        self.status = DataManagerStatus.STOPPING
        
        try:
            # Disconnect from all data sources
            for source_name, data_source in self.data_sources.items():
                data_source.disconnect()
            
            # Clear cache
            self.data_cache.clear()
            
            # Clear subscriptions
            self.subscriptions.clear()
            
            self.status = DataManagerStatus.STOPPED
            self.logger.info("Data manager stopped successfully")
            
            # Emit event
            self.event_bus.publish(Event(
                event_type=EventType.SYSTEM,
                data={"message": "Data manager stopped", "status": self.status.value}
            ))
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop data manager: {e}")
            self.status = DataManagerStatus.ERROR
            return False
    
    def get_historical_data(
        self,
        symbol: str,
        frequency: DataFrequency,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None,
        source_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get historical market data.
        
        Args:
            symbol: Trading symbol
            frequency: Data frequency
            start_time: Start time
            end_time: End time
            limit: Maximum number of data points
            source_name: Name of data source to use (if None, uses default)
            
        Returns:
            DataFrame with historical data
        """
        # Check cache first
        cache_key = f"{symbol}_{frequency.value}_{start_time}_{end_time}_{limit}"
        cached_data = self.data_cache.get(cache_key)
        if cached_data is not None and not cached_data.empty:
            return cached_data
        
        # Get data source
        data_source = self._get_data_source(source_name)
        if data_source is None:
            self.logger.error(f"Data source not found: {source_name}")
            return pd.DataFrame()
        
        try:
            # Get historical data
            data = data_source.get_historical_data(
                symbol, frequency, start_time, end_time, limit
            )
            
            # Cache the data
            if not data.empty:
                self.data_cache.put(cache_key, data)
            
            return data
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()
    
    def get_latest_data(
        self,
        symbol: str,
        frequency: DataFrequency,
        count: int = 1,
        source_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get the latest market data.
        
        Args:
            symbol: Trading symbol
            frequency: Data frequency
            count: Number of latest data points
            source_name: Name of data source to use (if None, uses default)
            
        Returns:
            DataFrame with latest data
        """
        # Get data source
        data_source = self._get_data_source(source_name)
        if data_source is None:
            self.logger.error(f"Data source not found: {source_name}")
            return pd.DataFrame()
        
        try:
            # Get latest data
            data = data_source.get_latest_data(symbol, frequency, count)
            return data
        except Exception as e:
            self.logger.error(f"Error getting latest data: {e}")
            return pd.DataFrame()
    
    def subscribe_to_realtime_data(
        self,
        symbols: List[str],
        frequency: DataFrequency,
        callback: Callable[[MarketData], None],
        source_name: Optional[str] = None
    ) -> bool:
        """
        Subscribe to real-time market data.
        
        Args:
            symbols: List of symbols to subscribe to
            frequency: Data frequency
            callback: Callback function for data updates
            source_name: Name of data source to use (if None, uses default)
            
        Returns:
            True if subscription successful, False otherwise
        """
        # Get data source
        data_source = self._get_data_source(source_name)
        if data_source is None:
            self.logger.error(f"Data source not found: {source_name}")
            return False
        
        try:
            # Subscribe to real-time data
            success = data_source.subscribe_to_realtime_data(symbols, frequency, callback)
            
            if success:
                # Store subscription
                for symbol in symbols:
                    self.subscriptions[symbol].append(callback)
                
                self.logger.info(f"Subscribed to real-time data for symbols: {symbols}")
            
            return success
        except Exception as e:
            self.logger.error(f"Error subscribing to real-time data: {e}")
            return False
    
    def unsubscribe_from_realtime_data(
        self,
        symbols: List[str],
        callback: Optional[Callable[[MarketData], None]] = None,
        source_name: Optional[str] = None
    ) -> bool:
        """
        Unsubscribe from real-time market data.
        
        Args:
            symbols: List of symbols to unsubscribe from
            callback: Specific callback to remove (if None, removes all callbacks for symbols)
            source_name: Name of data source to use (if None, uses default)
            
        Returns:
            True if unsubscription successful, False otherwise
        """
        # Get data source
        data_source = self._get_data_source(source_name)
        if data_source is None:
            self.logger.error(f"Data source not found: {source_name}")
            return False
        
        try:
            # Unsubscribe from real-time data
            success = data_source.unsubscribe_from_realtime_data(symbols)
            
            if success:
                # Remove subscription
                for symbol in symbols:
                    if symbol in self.subscriptions:
                        if callback is None:
                            # Remove all callbacks for this symbol
                            del self.subscriptions[symbol]
                        else:
                            # Remove specific callback
                            try:
                                self.subscriptions[symbol].remove(callback)
                                if not self.subscriptions[symbol]:
                                    del self.subscriptions[symbol]
                            except ValueError:
                                pass
                
                self.logger.info(f"Unsubscribed from real-time data for symbols: {symbols}")
            
            return success
        except Exception as e:
            self.logger.error(f"Error unsubscribing from real-time data: {e}")
            return False
    
    def is_data_available(
        self,
        symbol: str,
        frequency: DataFrequency,
        start_time: datetime,
        end_time: datetime,
        source_name: Optional[str] = None
    ) -> bool:
        """
        Check if data is available for the given parameters.
        
        Args:
            symbol: Trading symbol
            frequency: Data frequency
            start_time: Start time
            end_time: End time
            source_name: Name of data source to use (if None, uses default)
            
        Returns:
            True if data is available, False otherwise
        """
        # Get data source
        data_source = self._get_data_source(source_name)
        if data_source is None:
            self.logger.error(f"Data source not found: {source_name}")
            return False
        
        try:
            return data_source.is_data_available(symbol, frequency, start_time, end_time)
        except Exception as e:
            self.logger.error(f"Error checking data availability: {e}")
            return False
    
    def get_supported_symbols(self, source_name: Optional[str] = None) -> List[str]:
        """
        Get list of supported symbols.
        
        Args:
            source_name: Name of data source (if None, returns symbols from all sources)
            
        Returns:
            List of supported symbols
        """
        if source_name:
            data_source = self._get_data_source(source_name)
            if data_source is None:
                return []
            return data_source.get_supported_symbols()
        else:
            # Get symbols from all data sources
            symbols = set()
            for data_source in self.data_sources.values():
                symbols.update(data_source.get_supported_symbols())
            return list(symbols)
    
    def get_supported_frequencies(self, source_name: Optional[str] = None) -> List[DataFrequency]:
        """
        Get list of supported frequencies.
        
        Args:
            source_name: Name of data source (if None, returns frequencies from all sources)
            
        Returns:
            List of supported frequencies
        """
        if source_name:
            data_source = self._get_data_source(source_name)
            if data_source is None:
                return []
            return data_source.get_supported_frequencies()
        else:
            # Get frequencies from all data sources
            frequencies = set()
            for data_source in self.data_sources.values():
                frequencies.update(data_source.get_supported_frequencies())
            return list(frequencies)
    
    def _get_data_source(self, source_name: Optional[str] = None) -> Optional[DataSource]:
        """
        Get data source by name.
        
        Args:
            source_name: Name of data source (if None, returns default source)
            
        Returns:
            Data source or None if not found
        """
        if source_name is None:
            # Return the first available data source
            if self.data_sources:
                return next(iter(self.data_sources.values()))
            return None
        
        return self.data_sources.get(source_name)
    
    def get_data_source_names(self) -> List[str]:
        """
        Get list of data source names.
        
        Returns:
            List of data source names
        """
        return list(self.data_sources.keys())
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get data manager status.
        
        Returns:
            Status dictionary
        """
        return {
            "status": self.status.value,
            "data_sources": list(self.data_sources.keys()),
            "subscriptions": {symbol: len(callbacks) for symbol, callbacks in self.subscriptions.items()},
            "cache_size": len(self.data_cache.cache)
        }


# Global data manager instance
_data_manager: Optional[DataManager] = None


def get_data_manager() -> DataManager:
    """
    Get the global data manager instance.
    
    Returns:
        Global data manager instance
    """
    global _data_manager
    if _data_manager is None:
        _data_manager = DataManager()
    return _data_manager


def initialize_data_manager(config: Optional[Dict[str, Any]] = None) -> DataManager:
    """
    Initialize the global data manager.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized data manager
    """
    global _data_manager
    _data_manager = DataManager(config)
    return _data_manager