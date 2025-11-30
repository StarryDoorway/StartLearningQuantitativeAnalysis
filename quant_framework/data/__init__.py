"""
Data module for the quantitative trading framework.

This module provides data access, management, and processing capabilities
for the quantitative trading framework.
"""

from .data_sources import (
    DataSource,
    DataSourceType,
    DataFrequency,
    MarketData,
    CSVDataSource
)

from .data_manager import (
    DataManager,
    DataManagerStatus,
    get_data_manager,
    initialize_data_manager
)

from .data_processor import (
    DataProcessor,
    IndicatorType
)

__all__ = [
    # Data sources
    "DataSource",
    "DataSourceType",
    "DataFrequency",
    "MarketData",
    "CSVDataSource",
    
    # Data manager
    "DataManager",
    "DataManagerStatus",
    "get_data_manager",
    "initialize_data_manager",
    
    # Data processor
    "DataProcessor",
    "IndicatorType"
]