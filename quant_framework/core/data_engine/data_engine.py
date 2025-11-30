"""
Data engine module for the quantitative trading framework.

This module provides a unified interface for accessing and managing market data
from various sources, with support for real-time streaming and historical data.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ...utils.config_loader import get_config
from ..event_bus import get_event_bus, EventType, Event

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Enumeration of different data source types."""
    EXCHANGE_API = "exchange_api"
    DATABASE = "database"
    FILE = "file"
    WEBSOCKET = "websocket"
    CUSTOM = "custom"


class DataFrequency(Enum):
    """Enumeration of data frequencies."""
    TICK = "tick"
    SECOND = "1s"
    MINUTE = "1m"
    HOUR = "1h"
    DAY = "1d"
    WEEK = "1w"
    MONTH = "1M"


@dataclass
class MarketData:
    """
    Market data container.
    
    Attributes:
        symbol: Trading symbol
        timestamp: Data timestamp
        open: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        volume: Trading volume
        frequency: Data frequency
        source: Data source
        additional_data: Additional data fields
    """
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    frequency: DataFrequency
    source: str
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'frequency': self.frequency.value,
            'source': self.source,
            **self.additional_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketData':
        """Create from dictionary."""
        return cls(
            symbol=data['symbol'],
            timestamp=data['timestamp'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            volume=data['volume'],
            frequency=DataFrequency(data['frequency']),
            source=data['source'],
            additional_data={k: v for k, v in data.items() 
                           if k not in ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'frequency', 'source']}
        )


class DataProvider(ABC):
    """Abstract base class for data providers."""
    
    @abstractmethod
    def get_historical_data(self, symbol: str, frequency: DataFrequency, 
                           start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Get historical market data.
        
        Args:
            symbol: Trading symbol
            frequency: Data frequency
            start_time: Start time
            end_time: End time
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
    
    @abstractmethod
    def get_latest_data(self, symbol: str, frequency: DataFrequency, 
                       count: int = 1) -> pd.DataFrame:
        """
        Get latest market data.
        
        Args:
            symbol: Trading symbol
            frequency: Data frequency
            count: Number of data points to retrieve
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
    
    @abstractmethod
    def subscribe_realtime(self, symbol: str, frequency: DataFrequency, 
                          callback: Callable[[MarketData], None]) -> None:
        """
        Subscribe to real-time market data.
        
        Args:
            symbol: Trading symbol
            frequency: Data frequency
            callback: Callback function for data updates
        """
        pass
    
    @abstractmethod
    def unsubscribe_realtime(self, symbol: str, frequency: DataFrequency) -> None:
        """
        Unsubscribe from real-time market data.
        
        Args:
            symbol: Trading symbol
            frequency: Data frequency
        """
        pass


class ExchangeDataProvider(DataProvider):
    """Data provider implementation for exchange APIs."""
    
    def __init__(self, exchange_name: str, config: Dict[str, Any]):
        """
        Initialize the exchange data provider.
        
        Args:
            exchange_name: Name of the exchange
            config: Exchange configuration
        """
        self.exchange_name = exchange_name
        self.config = config
        self._client = None
        self._subscriptions = {}
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the exchange client."""
        try:
            # Import the appropriate client based on exchange name
            if self.exchange_name.lower() == "okx":
                from ...src.core.okx_client import OKXClient
                self._client = OKXClient(
                    api_key=self.config.get("api_key"),
                    secret_key=self.config.get("secret_key"),
                    passphrase=self.config.get("passphrase"),
                    sandbox=self.config.get("sandbox", False)
                )
            else:
                raise ValueError(f"Unsupported exchange: {self.exchange_name}")
            
            logger.info(f"Initialized {self.exchange_name} client")
        except Exception as e:
            logger.error(f"Failed to initialize {self.exchange_name} client: {str(e)}")
            raise
    
    def get_historical_data(self, symbol: str, frequency: DataFrequency, 
                           start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get historical market data from exchange."""
        if not self._client:
            raise RuntimeError(f"Exchange client not initialized for {self.exchange_name}")
        
        try:
            # Convert frequency to exchange-specific format
            timeframe = self._convert_frequency(frequency)
            
            # Get data from exchange
            data = self._client.get_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {str(e)}")
            raise
    
    def get_latest_data(self, symbol: str, frequency: DataFrequency, 
                       count: int = 1) -> pd.DataFrame:
        """Get latest market data from exchange."""
        if not self._client:
            raise RuntimeError(f"Exchange client not initialized for {self.exchange_name}")
        
        try:
            # Convert frequency to exchange-specific format
            timeframe = self._convert_frequency(frequency)
            
            # Get data from exchange
            data = self._client.get_latest_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=count
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Failed to get latest data for {symbol}: {str(e)}")
            raise
    
    def subscribe_realtime(self, symbol: str, frequency: DataFrequency, 
                          callback: Callable[[MarketData], None]) -> None:
        """Subscribe to real-time market data from exchange."""
        if not self._client:
            raise RuntimeError(f"Exchange client not initialized for {self.exchange_name}")
        
        try:
            # Convert frequency to exchange-specific format
            channel = self._convert_frequency_to_channel(frequency)
            
            # Store callback
            self._subscriptions[(symbol, frequency)] = callback
            
            # Subscribe to real-time data
            self._client.subscribe(
                channel=channel,
                symbol=symbol,
                callback=self._handle_realtime_data
            )
            
            logger.info(f"Subscribed to real-time data for {symbol} at {frequency.value}")
        except Exception as e:
            logger.error(f"Failed to subscribe to real-time data for {symbol}: {str(e)}")
            raise
    
    def unsubscribe_realtime(self, symbol: str, frequency: DataFrequency) -> None:
        """Unsubscribe from real-time market data."""
        if not self._client:
            raise RuntimeError(f"Exchange client not initialized for {self.exchange_name}")
        
        try:
            # Convert frequency to exchange-specific format
            channel = self._convert_frequency_to_channel(frequency)
            
            # Remove from subscriptions
            if (symbol, frequency) in self._subscriptions:
                del self._subscriptions[(symbol, frequency)]
            
            # Unsubscribe from real-time data
            self._client.unsubscribe(channel=channel, symbol=symbol)
            
            logger.info(f"Unsubscribed from real-time data for {symbol} at {frequency.value}")
        except Exception as e:
            logger.error(f"Failed to unsubscribe from real-time data for {symbol}: {str(e)}")
    
    def _handle_realtime_data(self, data: Dict[str, Any]) -> None:
        """Handle real-time data from exchange."""
        try:
            # Extract data fields
            symbol = data.get('symbol')
            timestamp = pd.to_datetime(data.get('timestamp'), unit='ms')
            open_price = float(data.get('open'))
            high_price = float(data.get('high'))
            low_price = float(data.get('low'))
            close_price = float(data.get('close'))
            volume = float(data.get('volume'))
            
            # Determine frequency
            frequency = self._determine_frequency(data)
            
            # Create MarketData object
            market_data = MarketData(
                symbol=symbol,
                timestamp=timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                frequency=frequency,
                source=self.exchange_name
            )
            
            # Call the appropriate callback
            callback = self._subscriptions.get((symbol, frequency))
            if callback:
                callback(market_data)
            
            # Publish event
            event_bus = get_event_bus()
            event = Event(
                event_type=EventType.MARKET_DATA,
                timestamp=time.time(),
                data=market_data.to_dict(),
                source=f"{self.exchange_name}_data_provider"
            )
            event_bus.publish(event)
            
        except Exception as e:
            logger.error(f"Error handling real-time data: {str(e)}")
    
    def _convert_frequency(self, frequency: DataFrequency) -> str:
        """Convert DataFrequency to exchange-specific timeframe."""
        mapping = {
            DataFrequency.SECOND: "1s",
            DataFrequency.MINUTE: "1m",
            DataFrequency.HOUR: "1h",
            DataFrequency.DAY: "1d",
            DataFrequency.WEEK: "1w",
            DataFrequency.MONTH: "1M"
        }
        return mapping.get(frequency, "1m")
    
    def _convert_frequency_to_channel(self, frequency: DataFrequency) -> str:
        """Convert DataFrequency to exchange-specific channel."""
        mapping = {
            DataFrequency.SECOND: "tick",
            DataFrequency.MINUTE: "1m",
            DataFrequency.HOUR: "1h",
            DataFrequency.DAY: "1d"
        }
        return mapping.get(frequency, "1m")
    
    def _determine_frequency(self, data: Dict[str, Any]) -> DataFrequency:
        """Determine frequency from data."""
        # This would be exchange-specific based on the channel or data format
        # For now, default to MINUTE
        return DataFrequency.MINUTE


class DataEngine:
    """
    Main data engine class that manages data providers and provides a unified interface.
    """
    
    def __init__(self):
        """Initialize the data engine."""
        self._providers: Dict[str, DataProvider] = {}
        self._subscriptions: Dict[str, Dict[str, Any]] = {}
        self._config = get_config()
        self._event_bus = get_event_bus()
        self._initialize_providers()
    
    def _initialize_providers(self) -> None:
        """Initialize data providers based on configuration."""
        data_config = self._config.get("data", {})
        providers_config = data_config.get("providers", {})
        
        for provider_name, provider_config in providers_config.items():
            try:
                provider_type = provider_config.get("type")
                
                if provider_type == "exchange":
                    exchange_name = provider_config.get("exchange")
                    provider = ExchangeDataProvider(exchange_name, provider_config)
                    self._providers[provider_name] = provider
                    logger.info(f"Initialized exchange data provider: {provider_name}")
                else:
                    logger.warning(f"Unsupported provider type: {provider_type}")
            except Exception as e:
                logger.error(f"Failed to initialize provider {provider_name}: {str(e)}")
    
    def get_historical_data(self, symbol: str, frequency: DataFrequency, 
                           start_time: datetime, end_time: datetime,
                           provider: Optional[str] = None) -> pd.DataFrame:
        """
        Get historical market data.
        
        Args:
            symbol: Trading symbol
            frequency: Data frequency
            start_time: Start time
            end_time: End time
            provider: Optional provider name, uses default if not specified
            
        Returns:
            DataFrame with OHLCV data
        """
        if provider and provider not in self._providers:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Use default provider if not specified
        if not provider:
            provider = self._get_default_provider()
        
        return self._providers[provider].get_historical_data(
            symbol, frequency, start_time, end_time
        )
    
    def get_latest_data(self, symbol: str, frequency: DataFrequency, 
                       count: int = 1, provider: Optional[str] = None) -> pd.DataFrame:
        """
        Get latest market data.
        
        Args:
            symbol: Trading symbol
            frequency: Data frequency
            count: Number of data points to retrieve
            provider: Optional provider name, uses default if not specified
            
        Returns:
            DataFrame with OHLCV data
        """
        if provider and provider not in self._providers:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Use default provider if not specified
        if not provider:
            provider = self._get_default_provider()
        
        return self._providers[provider].get_latest_data(
            symbol, frequency, count
        )
    
    def subscribe_realtime(self, symbol: str, frequency: DataFrequency, 
                          callback: Optional[Callable[[MarketData], None]] = None,
                          provider: Optional[str] = None) -> None:
        """
        Subscribe to real-time market data.
        
        Args:
            symbol: Trading symbol
            frequency: Data frequency
            callback: Optional callback function for data updates
            provider: Optional provider name, uses default if not specified
        """
        if provider and provider not in self._providers:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Use default provider if not specified
        if not provider:
            provider = self._get_default_provider()
        
        # Create default callback if not provided
        if not callback:
            callback = self._default_realtime_callback
        
        # Store subscription
        subscription_id = f"{provider}_{symbol}_{frequency.value}"
        self._subscriptions[subscription_id] = {
            "provider": provider,
            "symbol": symbol,
            "frequency": frequency,
            "callback": callback
        }
        
        # Subscribe to provider
        self._providers[provider].subscribe_realtime(symbol, frequency, callback)
    
    def unsubscribe_realtime(self, symbol: str, frequency: DataFrequency, 
                           provider: Optional[str] = None) -> None:
        """
        Unsubscribe from real-time market data.
        
        Args:
            symbol: Trading symbol
            frequency: Data frequency
            provider: Optional provider name, uses default if not specified
        """
        if provider and provider not in self._providers:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Use default provider if not specified
        if not provider:
            provider = self._get_default_provider()
        
        # Remove from subscriptions
        subscription_id = f"{provider}_{symbol}_{frequency.value}"
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
        
        # Unsubscribe from provider
        self._providers[provider].unsubscribe_realtime(symbol, frequency)
    
    def _get_default_provider(self) -> str:
        """Get the default data provider."""
        data_config = self._config.get("data", {})
        default_provider = data_config.get("default_provider")
        
        if not default_provider or default_provider not in self._providers:
            # Use the first available provider
            if self._providers:
                default_provider = next(iter(self._providers))
            else:
                raise RuntimeError("No data providers available")
        
        return default_provider
    
    def _default_realtime_callback(self, data: MarketData) -> None:
        """Default callback for real-time data."""
        # Publish event
        event = Event(
            event_type=EventType.MARKET_DATA,
            timestamp=time.time(),
            data=data.to_dict(),
            source="data_engine"
        )
        self._event_bus.publish(event)
    
    def get_available_symbols(self, provider: Optional[str] = None) -> List[str]:
        """
        Get list of available symbols from a provider.
        
        Args:
            provider: Optional provider name, uses default if not specified
            
        Returns:
            List of available symbols
        """
        if provider and provider not in self._providers:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Use default provider if not specified
        if not provider:
            provider = self._get_default_provider()
        
        # This would need to be implemented in the provider
        # For now, return an empty list
        return []
    
    def get_provider_status(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of a data provider.
        
        Args:
            provider: Optional provider name, uses default if not specified
            
        Returns:
            Dictionary with provider status
        """
        if provider and provider not in self._providers:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Use default provider if not specified
        if not provider:
            provider = self._get_default_provider()
        
        # This would need to be implemented in the provider
        # For now, return a basic status
        return {
            "provider": provider,
            "status": "active",
            "subscriptions": len([s for s in self._subscriptions.values() if s["provider"] == provider])
        }


# Global data engine instance
_data_engine = None


def get_data_engine() -> DataEngine:
    """
    Get the global data engine instance.
    
    Returns:
        DataEngine instance
    """
    global _data_engine
    
    if _data_engine is None:
        _data_engine = DataEngine()
    
    return _data_engine


def reset_data_engine() -> None:
    """Reset the global data engine instance."""
    global _data_engine
    _data_engine = None